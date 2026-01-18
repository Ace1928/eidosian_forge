import collections
import inspect
import warnings
from functools import wraps
import tree
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import regularizers
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.distribution import distribution_lib
from keras.src.layers import input_spec
from keras.src.metrics.metric import Metric
from keras.src.ops.operation import Operation
from keras.src.utils import python_utils
from keras.src.utils import summary_utils
from keras.src.utils import traceback_utils
from keras.src.utils import tracking
from keras.src.utils.shape_utils import map_shape_structure
def _build_by_run_for_single_pos_arg(self, input_shape):
    input_tensors = map_shape_structure(lambda s: backend.KerasTensor(s), input_shape)
    try:
        backend.compute_output_spec(self.call, input_tensors)
        return True
    except:
        return False