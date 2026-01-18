from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_v1
from tensorflow.python.keras.engine.base_layer import AddMetric
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _reset_build_compile_trackers(model):
    """Reset state trackers for model.

  Note that we do not actually zero out attributes such as optimizer,
  but instead rely on the expectation that all of the attrs will be
  over-written on calling build/compile/etc. This is somewhat fragile,
  insofar as we check elsewhere for the presence of these attributes as
  evidence of having been built/compiled/etc. Pending a better way to do this,
  we reset key attributes here to allow building and compiling.

  Args:
    model: the model that is being reset
  """
    model.built = False
    model.inputs = None
    model.outputs = None
    model._is_compiled = False
    if not ops.executing_eagerly_outside_functions():
        model._v1_compile_was_called = False
    model.optimizer = None