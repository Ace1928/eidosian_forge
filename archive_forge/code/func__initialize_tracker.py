import collections
import inspect
import warnings
from functools import wraps
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
from keras.src.utils import tree
@tracking.no_automatic_dependency_tracking
def _initialize_tracker(self):
    if hasattr(self, '_tracker'):
        return
    trainable_variables = []
    non_trainable_variables = []
    layers = []
    metrics = []
    seed_generators = []
    self._tracker = tracking.Tracker({'trainable_variables': (lambda x: isinstance(x, backend.Variable) and x.trainable, trainable_variables), 'non_trainable_variables': (lambda x: isinstance(x, backend.Variable) and (not x.trainable), non_trainable_variables), 'metrics': (lambda x: isinstance(x, Metric), metrics), 'layers': (lambda x: isinstance(x, Layer) and (not isinstance(x, Metric)), layers), 'seed_generators': (lambda x: isinstance(x, backend.random.SeedGenerator), seed_generators)}, exclusions={'non_trainable_variables': ['trainable_variables']})
    if backend.backend() == 'tensorflow':
        _self_setattr_tracking = getattr(self, '_self_setattr_tracking', True)
        self._self_setattr_tracking = False
    self._trainable_variables = trainable_variables
    self._non_trainable_variables = non_trainable_variables
    self._layers = layers
    self._metrics = metrics
    self._seed_generators = seed_generators
    if backend.backend() == 'tensorflow':
        self._self_setattr_tracking = _self_setattr_tracking