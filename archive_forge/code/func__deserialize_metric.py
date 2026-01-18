import collections
import copy
import os
from tensorflow.python.eager import def_function
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _deserialize_metric(metric_config):
    """Deserialize metrics, leaving special strings untouched."""
    from tensorflow.python.keras import metrics as metrics_module
    if metric_config in ['accuracy', 'acc', 'crossentropy', 'ce']:
        return metric_config
    return metrics_module.deserialize(metric_config)