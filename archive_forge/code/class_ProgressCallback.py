from collections import Counter
import json
import numpy as np
import os
import pickle
import tempfile
import time
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.callback import Callback
from ray._private.test_utils import safe_write_to_results_json
class ProgressCallback(Callback):

    def __init__(self):
        self.last_update = 0
        self.update_interval = 60

    def on_step_end(self, iteration, trials, **kwargs):
        if time.time() - self.last_update > self.update_interval:
            now = time.time()
            result = {'last_update': now, 'iteration': iteration, 'trial_states': dict(Counter([trial.status for trial in trials]))}
            safe_write_to_results_json(result, '/tmp/release_test_out.json')
            self.last_update = now