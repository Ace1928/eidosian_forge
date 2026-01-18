import platform
import warnings
import tree
from keras.src import backend
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src import optimizers
from keras.src.optimizers.loss_scale_optimizer import LossScaleOptimizer
from keras.src.saving import serialization_lib
from keras.src.trainers.compile_utils import CompileLoss
from keras.src.trainers.compile_utils import CompileMetrics
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import traceback_utils
from keras.src.utils import tracking
def _pythonify_logs(self, logs):
    result = {}
    for key, value in sorted(logs.items()):
        if isinstance(value, dict):
            result.update(self._pythonify_logs(value))
        else:
            try:
                value = float(value)
            except:
                pass
            result[key] = value
    return result