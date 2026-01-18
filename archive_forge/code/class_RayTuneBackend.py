from .integrations import (
from .trainer_utils import (
from .utils import logging
class RayTuneBackend(HyperParamSearchBackendBase):
    name = 'ray'
    pip_package = "'ray[tune]'"

    @staticmethod
    def is_available():
        return is_ray_tune_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_ray(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        return default_hp_space_ray(trial)