import collections
import hashlib
import os
import random
import threading
import warnings
from datetime import datetime
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
def _check_consecutive_failures(self):
    consecutive_failures = 0
    for trial_id in self.end_order:
        trial = self.trials[trial_id]
        if trial.status == trial_module.TrialStatus.FAILED:
            consecutive_failures += 1
        else:
            consecutive_failures = 0
        if consecutive_failures == self.max_consecutive_failed_trials:
            raise RuntimeError(f'Number of consecutive failures exceeded the limit of {self.max_consecutive_failed_trials}.\n' + (trial.message or ''))