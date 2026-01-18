import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import keras_tuner
import numpy as np
from autokeras.engine import tuner as tuner_module
def _get_best_hps(self):
    best_trials = self.get_best_trials()
    if best_trials:
        return best_trials[0].hyperparameters.copy()
    else:
        return self.hyperparameters.copy()