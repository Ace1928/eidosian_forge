import copy
import math
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import tuner as tuner_module
def _build_hypermodel(self, hp):
    model = super()._build_hypermodel(hp)
    if 'tuner/trial_id' in hp.values:
        trial_id = hp.values['tuner/trial_id']
        if backend.config.multi_backend():
            model.build_from_config(utils.load_json(self._get_build_config_fname(trial_id)))
        model.load_weights(self._get_checkpoint_fname(trial_id))
    return model