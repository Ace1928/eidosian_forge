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
class H5IOStore:

    def __init__(self, root_path, archive=None, mode='r'):
        """Numerical variable store backed by HDF5.

        If `archive` is specified, then `root_path` refers to the filename
        inside the archive.

        If `archive` is not specified, then `root_path` refers to the path of
        the h5 file on disk.
        """
        self.root_path = root_path
        self.mode = mode
        self.archive = archive
        self.io_file = None
        if self.archive:
            if self.mode == 'w':
                self.io_file = io.BytesIO()
            else:
                self.io_file = self.archive.open(self.root_path, 'r')
            self.h5_file = h5py.File(self.io_file, mode=self.mode)
        else:
            self.h5_file = h5py.File(root_path, mode=self.mode)

    def make(self, path):
        if not path:
            return self.h5_file.create_group('vars')
        return self.h5_file.create_group(path).create_group('vars')

    def get(self, path):
        if not path:
            return self.h5_file['vars']
        if path in self.h5_file and 'vars' in self.h5_file[path]:
            return self.h5_file[path]['vars']
        return {}

    def close(self):
        self.h5_file.close()
        if self.mode == 'w' and self.archive:
            self.archive.writestr(self.root_path, self.io_file.getvalue())
        if self.io_file:
            self.io_file.close()