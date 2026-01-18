import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.engine import base_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import io_utils
def _list_variables_used_by_fns(fns):
    trainable_variables = []
    non_trainable_variables = []
    trainable_variables_ids = set()
    non_trainable_variables_ids = set()
    for fn in fns:
        if hasattr(fn, 'concrete_functions'):
            concrete_functions = fn.concrete_functions
        elif hasattr(fn, 'get_concrete_function'):
            concrete_functions = [fn.get_concrete_function()]
        else:
            concrete_functions = [fn]
        for concrete_fn in concrete_functions:
            for v in concrete_fn.trainable_variables:
                if id(v) not in trainable_variables_ids:
                    trainable_variables.append(v)
                    trainable_variables_ids.add(id(v))
            for v in concrete_fn.variables:
                if id(v) not in trainable_variables_ids and id(v) not in non_trainable_variables_ids:
                    non_trainable_variables.append(v)
                    non_trainable_variables_ids.add(id(v))
    return (trainable_variables, non_trainable_variables)