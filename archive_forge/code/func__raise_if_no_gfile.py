import hashlib
import os
import pathlib
import re
import shutil
import tarfile
import urllib
import warnings
import zipfile
from urllib.request import urlretrieve
from keras.src.api_export import keras_export
from keras.src.backend import config
from keras.src.utils import io_utils
from keras.src.utils.module_utils import gfile
from keras.src.utils.progbar import Progbar
def _raise_if_no_gfile(path):
    raise ValueError(f'Handling remote paths requires installing TensorFlow (in order to use gfile). Received path: {path}')