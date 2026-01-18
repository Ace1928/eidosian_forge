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
class Display(stateful.Stateful):

    def __init__(self, oracle, verbose=1):
        self.verbose = verbose
        self.oracle = oracle
        self.col_width = 18
        self.search_start = None
        self.trial_start = {}
        self.trial_number = {}

    def get_state(self):
        return {'search_start': self.search_start.isoformat() if self.search_start is not None else self.search_start, 'trial_start': {key: value.isoformat() for key, value in self.trial_start.items()}, 'trial_number': self.trial_number}

    def set_state(self, state):
        self.search_start = datetime.fromisoformat(state['search_start']) if state['search_start'] is not None else state['search_start']
        self.trial_start = {key: datetime.fromisoformat(value) for key, value in state['trial_start'].items()}
        self.trial_number = state['trial_number']

    def on_trial_begin(self, trial):
        if self.verbose < 1:
            return
        start_time = datetime.now()
        self.trial_start[trial.trial_id] = start_time
        if self.search_start is None:
            self.search_start = start_time
        current_number = len(self.oracle.trials)
        self.trial_number[trial.trial_id] = current_number
        print()
        print(f'Search: Running Trial #{current_number}')
        print()
        self.show_hyperparameter_table(trial)
        print()

    def on_trial_end(self, trial):
        if self.verbose < 1:
            return
        utils.try_clear()
        time_taken_str = self.format_duration(datetime.now() - self.trial_start[trial.trial_id])
        print(f'Trial {self.trial_number[trial.trial_id]} Complete [{time_taken_str}]')
        if trial.score is not None:
            print(f'{self.oracle.objective.name}: {trial.score}')
        print()
        best_trials = self.oracle.get_best_trials()
        best_score = best_trials[0].score if len(best_trials) > 0 else None
        print(f'Best {self.oracle.objective.name} So Far: {best_score}')
        time_elapsed_str = self.format_duration(datetime.now() - self.search_start)
        print(f'Total elapsed time: {time_elapsed_str}')

    def show_hyperparameter_table(self, trial):
        template = '{{0:{0}}}|{{1:{0}}}|{{2}}'.format(self.col_width)
        best_trials = self.oracle.get_best_trials()
        best_trial = best_trials[0] if len(best_trials) > 0 else None
        if trial.hyperparameters.values:
            print(template.format('Value', 'Best Value So Far', 'Hyperparameter'))
            for hp, value in trial.hyperparameters.values.items():
                best_value = best_trial.hyperparameters.values.get(hp) if best_trial else '?'
                print(template.format(self.format_value(value), self.format_value(best_value), hp))
        else:
            print('default configuration')

    def format_value(self, val):
        if isinstance(val, (int, float)) and (not isinstance(val, bool)):
            return f'{val:.5g}'
        val_str = str(val)
        if len(val_str) > self.col_width:
            val_str = f'{val_str[:self.col_width - 3]}...'
        return val_str

    def format_duration(self, d):
        s = round(d.total_seconds())
        d = s // 86400
        s %= 86400
        h = s // 3600
        s %= 3600
        m = s // 60
        s %= 60
        if d > 0:
            return f'{d:d}d {h:02d}h {m:02d}m {s:02d}s'
        return f'{h:02d}h {m:02d}m {s:02d}s'