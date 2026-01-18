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
class DLProgbar:
    """Manage progress bar state for use in urlretrieve."""

    def __init__(self):
        self.progbar = None
        self.finished = False

    def __call__(self, block_num, block_size, total_size):
        if not self.progbar:
            if total_size == -1:
                total_size = None
            self.progbar = Progbar(total_size)
        current = block_num * block_size
        if total_size is None:
            self.progbar.update(current)
        elif current < total_size:
            self.progbar.update(current)
        elif not self.finished:
            self.progbar.update(self.progbar.target)
            self.finished = True