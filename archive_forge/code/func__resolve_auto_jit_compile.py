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
def _resolve_auto_jit_compile(self):
    if backend.backend() == 'torch':
        return False
    if backend.backend() == 'tensorflow':
        import tensorflow as tf
        devices = tf.config.list_physical_devices()
        if not list(filter(lambda x: x.device_type != 'CPU', devices)):
            return False
        if self._distribute_strategy:
            return False
    if model_supports_jit(self):
        return True
    return False