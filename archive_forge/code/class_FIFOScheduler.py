from typing import Dict, Optional, TYPE_CHECKING
from ray.air._internal.usage import tag_scheduler
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI
class FIFOScheduler(TrialScheduler):
    """Simple scheduler that just runs trials in submission order."""

    def __init__(self):
        super().__init__()

    def on_trial_add(self, tune_controller: 'TuneController', trial: Trial):
        pass

    def on_trial_error(self, tune_controller: 'TuneController', trial: Trial):
        pass

    def on_trial_result(self, tune_controller: 'TuneController', trial: Trial, result: Dict) -> str:
        return TrialScheduler.CONTINUE

    def on_trial_complete(self, tune_controller: 'TuneController', trial: Trial, result: Dict):
        pass

    def on_trial_remove(self, tune_controller: 'TuneController', trial: Trial):
        pass

    def choose_trial_to_run(self, tune_controller: 'TuneController') -> Optional[Trial]:
        for trial in tune_controller.get_trials():
            if trial.status == Trial.PENDING:
                return trial
        for trial in tune_controller.get_trials():
            if trial.status == Trial.PAUSED:
                return trial
        return None

    def debug_string(self) -> str:
        return 'Using FIFO scheduling algorithm.'