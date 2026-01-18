import functools
import hashlib
import multiprocessing.dummy
import os
import pathlib
import queue
import random
import shutil
import tarfile
import threading
import time
import typing
import urllib
import warnings
import weakref
import zipfile
from abc import abstractmethod
from contextlib import closing
import numpy as np
import tensorflow.compat.v2 as tf
from six.moves.urllib.parse import urlsplit
from keras.src.utils import io_utils
from keras.src.utils import tf_inspect
from keras.src.utils.generic_utils import Progbar
from tensorflow.python.util.tf_export import keras_export
from six.moves.urllib.request import urlopen
def _is_path_in_dir(path, base_dir):
    return _resolve_path(os.path.join(base_dir, path)).startswith(base_dir)