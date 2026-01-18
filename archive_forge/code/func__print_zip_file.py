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
def _print_zip_file(zipfile, action):
    io_utils.print_msg(f'Keras model archive {action}:')
    io_utils.print_msg('%-46s %19s %12s' % ('File Name', 'Modified    ', 'Size'))
    for zinfo in zipfile.filelist:
        date = '%d-%02d-%02d %02d:%02d:%02d' % zinfo.date_time[:6]
        io_utils.print_msg('%-46s %s %12d' % (zinfo.filename, date, zinfo.file_size))