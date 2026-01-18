import os
import numpy as np
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.data.experimental.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import nest
def _delete_ckpt(self):
    prefix = self._ckpt_path()
    pattern = prefix + '*'
    files = gfile.Glob(pattern)
    map(gfile.Remove, files)