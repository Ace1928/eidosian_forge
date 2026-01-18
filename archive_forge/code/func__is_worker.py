import copy
import os
import traceback
import warnings
from keras_tuner.src import backend
from keras_tuner.src import config as config_module
from keras_tuner.src import errors
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.distribute import utils as dist_utils
from keras_tuner.src.engine import hypermodel as hm_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner_utils
def _is_worker(self):
    """Return true only if in parallel tuning and is a worker tuner."""
    return dist_utils.has_chief_oracle() and (not dist_utils.is_chief_oracle())