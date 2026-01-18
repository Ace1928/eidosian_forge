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
def _get_own_losses(self):
    if backend.in_stateless_scope():
        losses = []
        scope = backend.get_stateless_scope()
        for loss in scope.losses:
            if id(loss) in self._loss_ids:
                losses.append(loss)
        return losses
    else:
        return self._losses[:]