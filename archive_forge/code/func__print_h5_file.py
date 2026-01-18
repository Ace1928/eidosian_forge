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
def _print_h5_file(h5_file, prefix='', action=None):
    if not prefix:
        print(f'Keras weights file ({h5_file}) {action}:')
    if not hasattr(h5_file, 'keys'):
        return
    for key in h5_file.keys():
        print(f'...{prefix}{key}')
        _print_h5_file(h5_file[key], prefix=prefix + '...')