import logging
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, List, Union
from ray.air.constants import TRAINING_ITERATION
from ray.tune.logger.logger import LoggerCallback
from ray.tune.result import (
from ray.tune.utils import flatten_dict
from ray.util.annotations import PublicAPI
def _log_trial_hparams(self, trial: 'Trial'):
    params = flatten_dict(trial.evaluated_params, delimiter='/')
    flat_params = flatten_dict(params)
    scrubbed_params = {k: v for k, v in flat_params.items() if isinstance(v, self.VALID_HPARAMS)}
    np_params = {k: v.tolist() for k, v in flat_params.items() if isinstance(v, self.VALID_NP_HPARAMS)}
    scrubbed_params.update(np_params)
    removed = {k: v for k, v in flat_params.items() if not isinstance(v, self.VALID_HPARAMS + self.VALID_NP_HPARAMS)}
    if removed:
        logger.info('Removed the following hyperparameter values when logging to aim: %s', str(removed))
    run = self._trial_to_run[trial]
    run['hparams'] = scrubbed_params