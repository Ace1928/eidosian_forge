import contextlib
import copy
from io import BytesIO
from typing import Any, Dict, List, Optional
import catalogue
import numpy
from ..backends import Ops, get_current_ops
from ..compat import cupy, h5py
from ..compat import tensorflow as tf
from ..optimizers import Optimizer
from ..types import ArgsKwargs, ArrayXd
from ..util import get_array_module
from .shim import Shim
class TensorFlowShim(Shim):
    """Interface between a TensorFlow model and a Thinc Model. This container is
    *not* a Thinc Model subclass itself.

    Reference for custom training:
    https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
    """
    gradients: Optional[List['tf.Tensor']]

    def __init__(self, model: Any, config=None, optimizer: Any=None):
        super().__init__(model, config, optimizer)
        self.gradients = None

    def __str__(self):
        lines: List[str] = []

        def accumulate(line: str):
            lines.append(line)
        self._model.summary(print_fn=accumulate)
        return '\n'.join(lines)

    def __call__(self, X: ArgsKwargs, is_train: bool):
        if is_train:
            return self.begin_update(X)
        else:
            return self.predict(X)

    def predict(self, X: ArgsKwargs):
        old_phase = tf.keras.backend.learning_phase()
        tf.keras.backend.set_learning_phase(0)
        Y = self._model(*X.args, **X.kwargs)
        tf.keras.backend.set_learning_phase(old_phase)
        return Y

    def begin_update(self, X: ArgsKwargs):
        tf.keras.backend.set_learning_phase(1)
        tape = tf.GradientTape()
        tape.__enter__()
        tape.watch(X.args)
        output = self._model(*X.args, **X.kwargs)

        def backprop(d_output):
            tape.__exit__(None, None, None)
            if len(X.args) == 1:
                wrt_tensors = [X.args[0]]
            else:
                wrt_tensors = list(X.args[0])
            wrt_tensors.extend(self._model.trainable_variables)
            all_gradients = tape.gradient(output, wrt_tensors, output_gradients=d_output)
            dX = all_gradients[:len(X.args)]
            opt_grads = all_gradients[1:]
            if self.gradients is not None:
                assert len(opt_grads) == len(self.gradients), 'gradients must match'
                variable: tf.Variable
                for variable, new_variable in zip(self.gradients, opt_grads):
                    variable.assign_add(new_variable)
            else:
                self.gradients = [tf.Variable(f) for f in opt_grads]
            return ArgsKwargs(args=tuple(dX), kwargs={})
        return (output, backprop)

    def finish_update(self, optimizer: Optimizer):
        if self.gradients is None:
            raise ValueError('There are no gradients for optimization. Be sure to call begin_update before calling finish_update.')
        assert len(self.gradients) == len(self._model.trainable_variables)
        grad: tf.Tensor
        variable: tf.Variable
        params = []
        grads = []
        shapes = []
        for grad, variable in zip(self.gradients, self._model.trainable_variables):
            param = variable.numpy()
            grad = grad.numpy()
            shapes.append((param.size, param.shape))
            params.append(param.ravel())
            grads.append(grad.ravel())
        xp = get_array_module(params[0])
        flat_params, flat_grads = optimizer((self.id, 'tensorflow-shim'), xp.concatenate(params), xp.concatenate(grads))
        start = 0
        for grad, variable in zip(self.gradients, self._model.trainable_variables):
            size, shape = shapes.pop(0)
            param = flat_params[start:start + size].reshape(shape)
            variable.assign(param)
            start += size
        self.gradients = None

    def _load_weights_from_state_dict(self, state_dict: Optional[Dict[str, ArrayXd]]=None):
        if state_dict is None:
            state_dict = self._create_state_dict()
        for layer in self._model.layers:
            current_layer_weights = []
            for weight in layer.weights:
                current_layer_weights.append(state_dict[weight.name])
            layer.set_weights(current_layer_weights)

    def _create_state_dict(self):
        state_dict = {}
        for layer in self._model.layers:
            for weight in layer.weights:
                state_dict[weight.name] = weight.numpy()
        return state_dict

    @contextlib.contextmanager
    def use_params(self, params):
        key_prefix = f'tensorflow_{self.id}_'
        state_dict = {}
        for k, v in params.items():
            if hasattr(k, 'startswith') and k.startswith(key_prefix):
                if cupy is None:
                    assert isinstance(v, numpy.ndarray)
                else:
                    if isinstance(v, cupy.core.core.ndarray):
                        v = cupy.asnumpy(v)
                    assert isinstance(v, numpy.ndarray)
                state_dict[k.replace(key_prefix, '')] = v
        if state_dict:
            backup = self._create_state_dict()
            self._load_weights_from_state_dict(state_dict)
            yield
            self._load_weights_from_state_dict(backup)
        else:
            yield

    def _clone_model(self):
        """similar to tf.keras.models.clone_model()
        But the tf.keras.models.clone_model changes the names of tf.Variables.
        This method even preserves that
        """
        model_json_config = self._model.to_json()
        tf.keras.backend.clear_session()
        self._model = tf.keras.models.model_from_json(model_json_config)
        self._load_weights_from_state_dict()

    def copy(self):
        model_json_config = self._model.to_json()
        self._model = None
        tf.keras.backend.clear_session()
        copied = copy.deepcopy(self)
        copied._model = tf.keras.models.model_from_json(model_json_config)
        copied._load_weights_from_state_dict()
        return copied

    def to_device(self, device_type: str, device_id: int):
        if device_type == 'cpu':
            with tf.device('/CPU'):
                self._clone_model()
        elif device_type == 'gpu':
            with tf.device('/GPU:{}'.format(device_id)):
                self._clone_model()

    def to_bytes(self):
        filelike = BytesIO()
        try:
            with h5py.File(filelike, 'w') as f:
                self._model.save(f, save_format='h5')
            return filelike.getvalue()
        except NotImplementedError:
            if not hasattr(self._model, 'catalogue_name'):
                raise ValueError("Couldn't serialize to h5, and model has no factory function for component serialization.")
        keras_model_fns.get(self._model.catalogue_name)
        return (self._model.catalogue_name, self._model.get_weights())

    def from_bytes(self, data):
        ops: Ops = get_current_ops()
        if ops.device_type == 'cpu':
            device = 'CPU'
        else:
            device = tf.test.gpu_device_name()
        if isinstance(data, (str, bytes)):
            tf.keras.backend.clear_session()
            filelike = BytesIO(data)
            filelike.seek(0)
            with h5py.File(filelike, 'r') as f:
                with tf.device(device):
                    self._model = tf.keras.models.load_model(f)
                return
        catalogue_name, model_weights = data
        if self._model is None:
            model_fn = keras_model_fns.get(catalogue_name)
            tf.keras.backend.clear_session()
            with tf.device(device):
                if hasattr(self._model, 'eg_args'):
                    ak: ArgsKwargs = self._model.eg_args
                    new_model = model_fn(*ak.args, **ak.kwargs)
                else:
                    new_model = model_fn()
            self._model_initialized = maybe_handshake_model(new_model)
        self._model.set_weights(model_weights)