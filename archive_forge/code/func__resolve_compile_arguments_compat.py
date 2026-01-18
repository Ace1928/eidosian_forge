import json
import threading
import tree
from absl import logging
from keras.src import backend
from keras.src import layers
from keras.src import losses
from keras.src import metrics as metrics_module
from keras.src import models
from keras.src import optimizers
from keras.src.legacy.saving import serialization
from keras.src.saving import object_registration
def _resolve_compile_arguments_compat(obj, obj_config, module):
    """Resolves backwards compatiblity issues with training config arguments.

    This helper function accepts built-in Keras modules such as optimizers,
    losses, and metrics to ensure an object being deserialized is compatible
    with Keras 3 built-ins. For legacy H5 files saved within Keras 3,
    this does nothing.
    """
    if isinstance(obj, str) and obj not in module.ALL_OBJECTS_DICT:
        obj = module.get(obj_config['config']['name'])
    return obj