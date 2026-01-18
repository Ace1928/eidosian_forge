import logging
from typing import Dict, Optional, TYPE_CHECKING
from ray.tune.schedulers.trial_scheduler import TrialScheduler
from ray.tune.schedulers.hyperband import HyperBandScheduler
from ray.tune.experiment import Trial
from ray.util import PublicAPI
def _unpause_trial(self, tune_controller: 'TuneController', trial: Trial):
    tune_controller.search_alg.searcher.on_unpause(trial.trial_id)