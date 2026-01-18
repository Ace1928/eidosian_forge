from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.eager import function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import func_graph
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
class _EstimatorWrappedGraph(wrap_function.WrappedGraph):
    """WrappedGraph that handles global step creation and wraps estimator fns."""

    def __init__(self, *args, **kwargs):
        super(_EstimatorWrappedGraph, self).__init__(*args, **kwargs)
        self._global_step_read_fn = self.wrap_function(self._global_step, signature=[])
        self._concrete_model_fn = None
        self._estimator_spec = None

    def _global_step(self):
        return tf.compat.v1.train.get_or_create_global_step()

    @property
    def global_step(self):
        return self._global_step_read_fn()

    @property
    def model_fn(self):
        return self._concrete_model_fn

    @property
    def estimator_spec(self):
        if self._concrete_model_fn is None:
            raise ValueError('Please wrap a model function first.')
        return self._estimator_spec

    def wrap_model_fn(self, model_fn, mode, args=None, kwargs=None, signature=None):
        """Wraps a model function, and stores the returned estimator spec."""
        if self._concrete_model_fn is not None:
            raise ValueError('`wrap_model_fn` should be only called once per graph.')

        def fn(*args, **kwargs):
            """Returns tensor and op outputs from the returned spec."""
            ret = model_fn(*args, **kwargs)
            if isinstance(ret, model_fn_lib.EstimatorSpec):
                self._estimator_spec = ret
                return _filter_estimator_spec_outputs(ret)
            return ret
        name = 'model_fn_{}'.format(mode)
        self._concrete_model_fn = self._wrap_function(fn, args, kwargs, signature, name)
        return self._concrete_model_fn

    def wrap_input_receiver_fn(self, input_receiver_fn):
        """Converts an input receiver function to one or more concrete functions.

    Input receiver functions are python functions with no arguments.
    Placeholders are created within the function and used to receive inputs to
    the model.

    The function (or multiple functions) generated depends on the InputReceiver
    object returned by `input_receiver_fn`.

    Generally, the returned function will have inputs and outputs:
      input_receiver(**receiver_tensors) --> features

    or (if the InputReceiver returns labels):
      input_receiver(**receiver_tensors) --> features, labels

    __Alternate Receiver Tensors__

    The InputReceiver may have alternate receiver tensors, in which case
    additional concrete functions are generated. Example:
      InputReceiver.receiver_tensors_alternatives = {
        'alt_input_1': Tensor,
        'alt_input_2': {
          'tensor_1': Tensor,
          'tensor_2': Tensor
        }
      }

    This will generate concrete functions:
      input_receiver_alt_input_1(input) --> features
      input_receiver_alt_input_2(tensor_1, tensor_2) --> features

    Args:
      input_receiver_fn: a no-argument function that returns an `InputReceiver`
        object.

    Returns:
      A list of tuples of (concrete function, receiver name). The name of the
      default input receiver is `None`.
    """
        ret = [None]

        def fn():
            ret[0] = input_receiver = input_receiver_fn()
            features = input_receiver.features
            labels = getattr(input_receiver, 'labels', None)
            if labels is None:
                return features
            return (features, labels)
        func_graph.func_graph_from_py_func(None, self._variable_holder.call_with_variable_creator_scope(fn), args=None, kwargs=None, signature=[], add_control_dependencies=False, func_graph=self.graph)
        functions = []
        input_receiver = ret[0]
        wrapped_input_receiver_fn = _prune_receiver_tensors(self._wrapped_function, receiver_tensors=input_receiver.receiver_tensors, outputs=self.graph.structured_outputs, name=_input_receiver_fn_name(None))
        functions.append((wrapped_input_receiver_fn, None))
        receiver_tensors_alternatives = getattr(input_receiver, 'receiver_tensors_alternatives', None)
        if receiver_tensors_alternatives:
            for receiver_name, receiver_tensors_alt in six.iteritems(receiver_tensors_alternatives):
                receiver_tensors_alt = _canonicalize_receiver_tensors(receiver_tensors_alt)
                wrapped_input_receiver_fn = _prune_receiver_tensors(self._wrapped_function, receiver_tensors=receiver_tensors_alt, outputs=self.graph.structured_outputs, name=_input_receiver_fn_name(receiver_name))
                functions.append((wrapped_input_receiver_fn, receiver_name))
        return functions