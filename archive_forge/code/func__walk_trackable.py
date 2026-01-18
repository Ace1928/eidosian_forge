import datetime
import io
import json
import os
import re
import tempfile
import threading
import warnings
import zipfile
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src import losses
from keras.src.engine import base_layer
from keras.src.optimizers import optimizer
from keras.src.saving.serialization_lib import ObjectSharingScope
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from keras.src.utils import generic_utils
from keras.src.utils import io_utils
def _walk_trackable(trackable):
    for child_attr in dir(trackable):
        if child_attr.startswith('__') or child_attr in ATTR_SKIPLIST:
            continue
        try:
            child_obj = getattr(trackable, child_attr)
        except Exception:
            continue
        yield (child_attr, child_obj)