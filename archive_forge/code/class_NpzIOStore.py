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
class NpzIOStore:

    def __init__(self, root_path, archive=None, mode='r'):
        """Numerical variable store backed by NumPy.savez/load.

         If `archive` is specified, then `root_path` refers to the filename
        inside the archive.

        If `archive` is not specified, then `root_path` refers to the path of
        the npz file on disk.
        """
        self.root_path = root_path
        self.mode = mode
        self.archive = archive
        if mode == 'w':
            self.contents = {}
        else:
            if self.archive:
                self.f = archive.open(root_path, mode='r')
            else:
                self.f = open(root_path, mode='rb')
            self.contents = np.load(self.f, allow_pickle=True)

    def make(self, path):
        if not path:
            self.contents['__root__'] = {}
            return self.contents['__root__']
        self.contents[path] = {}
        return self.contents[path]

    def get(self, path):
        if not path:
            if '__root__' in self.contents:
                return dict(self.contents['__root__'])
            return {}
        if path in self.contents:
            return self.contents[path].tolist()
        return {}

    def close(self):
        if self.mode == 'w':
            if self.archive:
                self.f = self.archive.open(self.root_path, mode='w', force_zip64=True)
            else:
                self.f = open(self.root_path, mode='wb')
            np.savez(self.f, **self.contents)
        self.f.close()