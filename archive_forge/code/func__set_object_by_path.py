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
def _set_object_by_path(object_to_set, path, value):
    """Set the attribute of instance to the object.

    Args:
      object_to_set: the instance whose attribute should be set.
      path: the tuple/list of string and ints, representing the attribute names.
        Int means that the attribute to set is a item a list.
      value: the value of the attribute.
    """
    for i, attr_name in enumerate(path):
        if i == len(path) - 1:
            if isinstance(attr_name, int):
                object_to_set[attr_name] = value
            else:
                setattr(object_to_set, attr_name, value)
        elif isinstance(attr_name, int):
            object_to_set = object_to_set[attr_name]
        else:
            object_to_set = getattr(object_to_set, attr_name)