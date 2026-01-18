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
def _update_trackable_reference(model, lazy_init_variable_to_tf_variable_map):
    """Update the trackable object references for the model.

    Note that this method is only needed because of a corner case for model
    checkpoint, where it could accidently catch a LazyInitVariable in checkpoint
    dependency and not visible to the model attribute graph itself.

    Args:
      model: the keras model instance whose checkpoint dependency will be
        examed.
      lazy_init_variable_to_tf_variable_map: the dict between LazyInitVariable
        ID and newly created DVariable.
    """
    object_graph = tf.__internal__.tracking.ObjectGraphView(model)
    trackables, _ = object_graph.breadth_first_traversal()
    for trackable in trackables:
        for ref_name, ref in trackable._trackable_children().items():
            if _is_lazy_init_variable(ref):
                trackable._track_trackable(lazy_init_variable_to_tf_variable_map[id(ref)], ref_name, overwrite=True)