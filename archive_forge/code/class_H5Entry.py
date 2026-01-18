import datetime
import io
import json
import tempfile
import warnings
import zipfile
import ml_dtypes
import numpy as np
from keras.src import backend
from keras.src.backend.common import global_state
from keras.src.layers.layer import Layer
from keras.src.losses.loss import Loss
from keras.src.metrics.metric import Metric
from keras.src.optimizers.optimizer import Optimizer
from keras.src.saving.serialization_lib import ObjectSharingScope
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from keras.src.trainers.compile_utils import CompileMetrics
from keras.src.utils import file_utils
from keras.src.utils import naming
from keras.src.version import __version__ as keras_version
class H5Entry:
    """Leaf entry in a H5IOStore."""

    def __init__(self, h5_file, path, mode):
        self.h5_file = h5_file
        self.path = path
        self.mode = mode
        if mode == 'w':
            if not path:
                self.group = self.h5_file.create_group('vars')
            else:
                self.group = self.h5_file.create_group(self.path).create_group('vars')
        else:
            found = False
            if not path:
                self.group = self.h5_file['vars']
                found = True
            elif path in self.h5_file and 'vars' in self.h5_file[path]:
                self.group = self.h5_file[path]['vars']
                found = True
            elif '_layer_checkpoint_dependencies' in self.h5_file:
                path = path.replace('layers', '_layer_checkpoint_dependencies')
                self.path = path
                if path in self.h5_file and 'vars' in self.h5_file[path]:
                    self.group = self.h5_file[path]['vars']
                    found = True
            if not found:
                self.group = {}

    def __len__(self):
        return self.group.__len__()

    def keys(self):
        return self.group.keys()

    def items(self):
        return self.group.items()

    def values(self):
        return self.group.values()

    def __setitem__(self, key, value):
        if self.mode != 'w':
            raise ValueError('Setting a value is only allowed in write mode.')
        value = backend.convert_to_numpy(value)
        if backend.standardize_dtype(value.dtype) == 'bfloat16':
            ds = self.group.create_dataset(key, data=value)
            ds.attrs['dtype'] = 'bfloat16'
        else:
            self.group[key] = value

    def __getitem__(self, name):
        value = self.group[name]
        if 'dtype' in value.attrs and value.attrs['dtype'] == 'bfloat16':
            value = np.array(value, dtype=ml_dtypes.bfloat16)
        return value