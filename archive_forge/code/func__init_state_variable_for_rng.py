import collections
import contextlib
import re
import threading
import tensorflow.compat.v2 as tf
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.dtensor import lazy_variable
from keras.src.dtensor import utils
from keras.src.engine import base_layer
from tensorflow.python.util.tf_export import keras_export
def _init_state_variable_for_rng(model, layout_map):
    """Init the state variable in tf.ranodm.Generator.

    Since the BaseRandomLayer in keras explicitly untrack the
    tf.random.Generator, the variable in it will stay as LazyInitVariable, which
    cause runtime error if we don't replace them with proper DVariable. Since
    user usually are not aware the existence of those variable, we will just
    give them replicated layout since they are tiny.

    Args:
      model: the model whose layers will be checked to find the
        BaseRandomLayers.
      layout_map: used to get the default mesh information to create DVariable.
    """
    for l in model._flatten(predicate=lambda o: isinstance(o, base_layer.BaseRandomLayer)):
        keras_generator = l._random_generator
        if keras_generator._built and keras_generator._generator is None:
            raise ValueError('Keras is expected to use tf.random.Generator when using DTensor API. Please call `tf.keras.backend.experimental.enable_tf_random_generator` at the beginning of your program.')
        if hasattr(keras_generator, '_generator') and _is_lazy_init_variable(keras_generator._generator._state_var):
            keras_generator._generator._state_var = _create_dvariable(layout_map, '', keras_generator._generator._state_var)
        else:
            with dtensor.default_mesh(layout_map.get_default_mesh()):
                keras_generator._maybe_init()