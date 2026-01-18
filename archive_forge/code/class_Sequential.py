import copy
import warnings
from tensorflow.python import tf2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.module import module
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
class Sequential(functional.Functional):
    """`Sequential` groups a linear stack of layers into a `tf.keras.Model`.

  `Sequential` provides training and inference features on this model.

  Examples:

  >>> # Optionally, the first layer can receive an `input_shape` argument:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
  >>> # Afterwards, we do automatic shape inference:
  >>> model.add(tf.keras.layers.Dense(4))

  >>> # This is identical to the following:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.Input(shape=(16,)))
  >>> model.add(tf.keras.layers.Dense(8))

  >>> # Note that you can also omit the `input_shape` argument.
  >>> # In that case the model doesn't have any weights until the first call
  >>> # to a training/evaluation method (since it isn't yet built):
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8))
  >>> model.add(tf.keras.layers.Dense(4))
  >>> # model.weights not created yet

  >>> # Whereas if you specify the input shape, the model gets built
  >>> # continuously as you are adding layers:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
  >>> model.add(tf.keras.layers.Dense(4))
  >>> len(model.weights)
  4

  >>> # When using the delayed-build pattern (no input shape specified), you can
  >>> # choose to manually build your model by calling
  >>> # `build(batch_input_shape)`:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8))
  >>> model.add(tf.keras.layers.Dense(4))
  >>> model.build((None, 16))
  >>> len(model.weights)
  4

  ```python
  # Note that when using the delayed-build pattern (no input shape specified),
  # the model gets built the first time you call `fit`, `eval`, or `predict`,
  # or the first time you call the model on some input data.
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(8))
  model.add(tf.keras.layers.Dense(1))
  model.compile(optimizer='sgd', loss='mse')
  # This builds the model for the first time:
  model.fit(x, y, batch_size=32, epochs=10)
  ```
  """

    @trackable.no_automatic_dependency_tracking
    def __init__(self, layers=None, name=None):
        """Creates a `Sequential` model instance.

    Args:
      layers: Optional list of layers to add to the model.
      name: Optional name for the model.
    """
        super(functional.Functional, self).__init__(name=name, autocast=False)
        self.supports_masking = True
        self._compute_output_and_mask_jointly = True
        self._auto_track_sub_layers = False
        self._inferred_input_shape = None
        self._has_explicit_input_shape = False
        self._input_dtype = None
        self._layer_call_argspecs = {}
        self._created_nodes = set()
        self._graph_initialized = False
        self._use_legacy_deferred_behavior = False
        if layers:
            if not isinstance(layers, (list, tuple)):
                layers = [layers]
            for layer in layers:
                self.add(layer)

    @property
    def layers(self):
        layers = super(Sequential, self).layers
        if layers and isinstance(layers[0], input_layer.InputLayer):
            return layers[1:]
        return layers[:]

    @trackable.no_automatic_dependency_tracking
    def add(self, layer):
        """Adds a layer instance on top of the layer stack.

    Args:
        layer: layer instance.

    Raises:
        TypeError: If `layer` is not a layer instance.
        ValueError: In case the `layer` argument does not
            know its input shape.
        ValueError: In case the `layer` argument has
            multiple output tensors, or is already connected
            somewhere else (forbidden in `Sequential` models).
    """
        if hasattr(layer, '_keras_history'):
            origin_layer = layer._keras_history[0]
            if isinstance(origin_layer, input_layer.InputLayer):
                layer = origin_layer
                logging.warning('Please add `keras.layers.InputLayer` instead of `keras.Input` to Sequential model. `keras.Input` is intended to be used by Functional model.')
        if isinstance(layer, module.Module):
            if not isinstance(layer, base_layer.Layer):
                layer = functional.ModuleWrapper(layer)
        else:
            raise TypeError('The added layer must be an instance of class Layer. Found: ' + str(layer))
        tf_utils.assert_no_legacy_layers([layer])
        if not self._is_layer_name_unique(layer):
            raise ValueError('All layers added to a Sequential model should have unique names. Name "%s" is already the name of a layer in this model. Update the `name` argument to pass a unique name.' % (layer.name,))
        self.built = False
        set_inputs = False
        self._maybe_create_attribute('_self_tracked_trackables', [])
        if not self._self_tracked_trackables:
            if isinstance(layer, input_layer.InputLayer):
                set_inputs = True
            else:
                batch_shape, dtype = training_utils.get_input_shape_and_dtype(layer)
                if batch_shape:
                    x = input_layer.Input(batch_shape=batch_shape, dtype=dtype, name=layer.name + '_input')
                    layer(x)
                    set_inputs = True
            if set_inputs:
                outputs = nest.flatten(layer._inbound_nodes[-1].outputs)
                if len(outputs) != 1:
                    raise ValueError(SINGLE_LAYER_OUTPUT_ERROR_MSG)
                self.outputs = outputs
                self.inputs = layer_utils.get_source_inputs(self.outputs[0])
                self.built = True
                self._has_explicit_input_shape = True
        elif self.outputs:
            output_tensor = layer(self.outputs[0])
            if len(nest.flatten(output_tensor)) != 1:
                raise ValueError(SINGLE_LAYER_OUTPUT_ERROR_MSG)
            self.outputs = [output_tensor]
            self.built = True
        if set_inputs or self._graph_initialized:
            self._init_graph_network(self.inputs, self.outputs)
            self._graph_initialized = True
        else:
            self._self_tracked_trackables.append(layer)
            self._handle_deferred_layer_dependencies([layer])
        self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)

    @trackable.no_automatic_dependency_tracking
    def pop(self):
        """Removes the last layer in the model.

    Raises:
        TypeError: if there are no layers in the model.
    """
        if not self.layers:
            raise TypeError('There are no layers in the model.')
        layer = self._self_tracked_trackables.pop()
        self._layer_call_argspecs.pop(layer)
        if not self.layers:
            self.outputs = None
            self.inputs = None
            self.built = False
            self._inferred_input_shape = None
            self._has_explicit_input_shape = False
            self._graph_initialized = False
        elif self._graph_initialized:
            self.layers[-1]._outbound_nodes = []
            self.outputs = [self.layers[-1].output]
            self._init_graph_network(self.inputs, self.outputs)
            self.built = True

    @trackable.no_automatic_dependency_tracking
    def _build_graph_network_for_inferred_shape(self, input_shape, input_dtype=None):
        if input_shape is None or not self.layers:
            return
        if not tf2.enabled() or not ops.executing_eagerly_outside_functions():
            return
        if not self._has_explicit_input_shape and (not self._use_legacy_deferred_behavior):
            input_shape = tuple(input_shape)
            if self._inferred_input_shape is None:
                new_shape = input_shape
            else:
                new_shape = relax_input_shape(self._inferred_input_shape, input_shape)
            if new_shape is not None and new_shape != self._inferred_input_shape:
                with ops.init_scope():
                    inputs = input_layer.Input(batch_shape=new_shape, dtype=input_dtype, name=self.layers[0].name + '_input')
                    layer_input = inputs
                    created_nodes = set()
                    for layer in self.layers:
                        clear_previously_created_nodes(layer, self._created_nodes)
                        try:
                            layer_output = layer(layer_input)
                        except:
                            self._use_legacy_deferred_behavior = True
                            return
                        if len(nest.flatten(layer_output)) != 1:
                            raise ValueError(SINGLE_LAYER_OUTPUT_ERROR_MSG)
                        track_nodes_created_by_last_call(layer, created_nodes)
                        layer_input = layer_output
                        outputs = layer_output
                    self._created_nodes = created_nodes
                    try:
                        self._init_graph_network(inputs, outputs)
                        self._graph_initialized = True
                    except:
                        self._use_legacy_deferred_behavior = True
                self._inferred_input_shape = new_shape

    @generic_utils.default
    def build(self, input_shape=None):
        if self._graph_initialized:
            self._init_graph_network(self.inputs, self.outputs)
        else:
            if input_shape is None:
                raise ValueError('You must provide an `input_shape` argument.')
            self._build_graph_network_for_inferred_shape(input_shape)
            if not self.built:
                input_shape = tuple(input_shape)
                self._build_input_shape = input_shape
                super(Sequential, self).build(input_shape)
        self.built = True

    def call(self, inputs, training=None, mask=None):
        if not self._has_explicit_input_shape:
            if not tensor_util.is_tf_type(inputs) and (not isinstance(inputs, np_arrays.ndarray)):
                self._use_legacy_deferred_behavior = True
                self._build_input_shape = nest.map_structure(_get_shape_tuple, inputs)
                if tf2.enabled():
                    logging.warning('Layers in a Sequential model should only have a single input tensor, but we receive a %s input: %s\nConsider rewriting this model with the Functional API.' % (type(inputs), inputs))
            else:
                self._build_graph_network_for_inferred_shape(inputs.shape, inputs.dtype)
        if self._graph_initialized:
            if not self.built:
                self._init_graph_network(self.inputs, self.outputs)
            return super(Sequential, self).call(inputs, training=training, mask=mask)
        outputs = inputs
        for layer in self.layers:
            kwargs = {}
            argspec = self._layer_call_argspecs[layer].args
            if 'mask' in argspec:
                kwargs['mask'] = mask
            if 'training' in argspec:
                kwargs['training'] = training
            outputs = layer(inputs, **kwargs)
            if len(nest.flatten(outputs)) != 1:
                raise ValueError(SINGLE_LAYER_OUTPUT_ERROR_MSG)
            inputs = outputs
            mask = getattr(outputs, '_keras_mask', None)
        return outputs

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            shape = layer.compute_output_shape(shape)
        return shape

    def compute_mask(self, inputs, mask):
        outputs = self.call(inputs, mask=mask)
        return getattr(outputs, '_keras_mask', None)

    def predict_proba(self, x, batch_size=32, verbose=0):
        """Generates class probability predictions for the input samples.

    The input samples are processed batch by batch.

    Args:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    Returns:
        A Numpy array of probability predictions.
    """
        warnings.warn('`model.predict_proba()` is deprecated and will be removed after 2021-01-01. Please use `model.predict()` instead.')
        preds = self.predict(x, batch_size, verbose)
        if preds.min() < 0.0 or preds.max() > 1.0:
            logging.warning('Network returning invalid probability values. The last layer might not normalize predictions into probabilities (like softmax or sigmoid would).')
        return preds

    def predict_classes(self, x, batch_size=32, verbose=0):
        """Generate class predictions for the input samples.

    The input samples are processed batch by batch.

    Args:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    Returns:
        A numpy array of class predictions.
    """
        warnings.warn('`model.predict_classes()` is deprecated and will be removed after 2021-01-01. Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).')
        proba = self.predict(x, batch_size=batch_size, verbose=verbose)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    def get_config(self):
        layer_configs = []
        for layer in super(Sequential, self).layers:
            layer_configs.append(generic_utils.serialize_keras_object(layer))
        config = {'name': self.name, 'layers': copy.deepcopy(layer_configs)}
        if not self._is_graph_network and self._build_input_shape is not None:
            config['build_input_shape'] = self._build_input_shape
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if 'name' in config:
            name = config['name']
            build_input_shape = config.get('build_input_shape')
            layer_configs = config['layers']
        else:
            name = None
            build_input_shape = None
            layer_configs = config
        model = cls(name=name)
        for layer_config in layer_configs:
            layer = layer_module.deserialize(layer_config, custom_objects=custom_objects)
            model.add(layer)
        if not model.inputs and build_input_shape and isinstance(build_input_shape, (tuple, list)):
            model.build(build_input_shape)
        return model

    @property
    def input_spec(self):
        if hasattr(self, '_manual_input_spec'):
            return self._manual_input_spec
        if self.layers and hasattr(self.layers[0], 'input_spec'):
            return self.layers[0].input_spec
        return None

    @input_spec.setter
    def input_spec(self, value):
        self._manual_input_spec = value

    @property
    def _trackable_saved_model_saver(self):
        return model_serialization.SequentialSavedModelSaver(self)

    def _is_layer_name_unique(self, layer):
        for ref_layer in self.layers:
            if layer.name == ref_layer.name and ref_layer is not layer:
                return False
        return True

    def _assert_weights_created(self):
        if self._graph_initialized:
            return
        super(functional.Functional, self)._assert_weights_created()