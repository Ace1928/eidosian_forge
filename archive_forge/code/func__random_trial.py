import copy
import math
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import tuner as tuner_module
def _random_trial(self, trial_id, bracket):
    bracket_num = bracket['bracket_num']
    rounds = bracket['rounds']
    values = self._random_values()
    if values:
        values['tuner/epochs'] = self._get_epochs(bracket_num, 0)
        values['tuner/initial_epoch'] = 0
        values['tuner/bracket'] = self._current_bracket
        values['tuner/round'] = 0
        rounds[0].append({'past_id': None, 'id': trial_id})
        return {'status': 'RUNNING', 'values': values}
    elif self.ongoing_trials:
        return {'status': 'IDLE'}
    else:
        return {'status': 'STOPPED'}