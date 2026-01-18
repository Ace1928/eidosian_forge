import contextlib
import copy
import gc
import math
import os
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import errors
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import config
from keras_tuner.src.backend import keras
from keras_tuner.src.engine import base_tuner
from keras_tuner.src.engine import tuner_utils
def _get_tensorboard_dir(self, logdir, trial_id, execution):
    return os.path.join(str(logdir), str(trial_id), f'execution{str(execution)}')