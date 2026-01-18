import numpy as np
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner as tuner_module
def _random_populate_space(self):
    values = self._random_values()
    if values is None:
        return {'status': trial_module.TrialStatus.STOPPED, 'values': None}
    return {'status': trial_module.TrialStatus.RUNNING, 'values': values}