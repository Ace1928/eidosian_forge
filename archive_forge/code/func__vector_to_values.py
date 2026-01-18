import numpy as np
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner as tuner_module
def _vector_to_values(self, vector):
    hps = hp_module.HyperParameters()
    vector_index = 0
    for hp in self.hyperparameters.space:
        hps.merge([hp])
        if isinstance(hp, hp_module.Fixed):
            value = hp.value
        else:
            prob = vector[vector_index]
            vector_index += 1
            value = hp.prob_to_value(prob)
        if hps.is_active(hp):
            hps.values[hp.name] = value
    return hps.values