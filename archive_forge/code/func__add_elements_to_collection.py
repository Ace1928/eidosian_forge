import copy
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.legacy_tf_layers import variable_scope_shim
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
def _add_elements_to_collection(elements, collection_list):
    if context.executing_eagerly():
        raise RuntimeError('Using collections from Layers not supported in Eager mode. Tried to add %s to %s' % (elements, collection_list))
    elements = nest.flatten(elements)
    collection_list = nest.flatten(collection_list)
    for name in collection_list:
        collection = ops.get_collection_ref(name)
        collection_set = {id(e) for e in collection}
        for element in elements:
            if id(element) not in collection_set:
                collection.append(element)