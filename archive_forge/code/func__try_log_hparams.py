import logging
import numpy as np
from typing import TYPE_CHECKING, Dict
from ray.air.constants import TRAINING_ITERATION
from ray.tune.logger.logger import _LOGGER_DEPRECATION_WARNING, Logger, LoggerCallback
from ray.util.debug import log_once
from ray.tune.result import (
from ray.tune.utils import flatten_dict
from ray.util.annotations import Deprecated, PublicAPI
def _try_log_hparams(self, trial: 'Trial', result: Dict):
    flat_params = flatten_dict(trial.evaluated_params)
    scrubbed_params = {k: v for k, v in flat_params.items() if isinstance(v, self.VALID_HPARAMS)}
    np_params = {k: v.tolist() for k, v in flat_params.items() if isinstance(v, self.VALID_NP_HPARAMS)}
    scrubbed_params.update(np_params)
    removed = {k: v for k, v in flat_params.items() if not isinstance(v, self.VALID_HPARAMS + self.VALID_NP_HPARAMS)}
    if removed:
        logger.info('Removed the following hyperparameter values when logging to tensorboard: %s', str(removed))
    from tensorboardX.summary import hparams
    try:
        experiment_tag, session_start_tag, session_end_tag = hparams(hparam_dict=scrubbed_params, metric_dict=result)
        self._trial_writer[trial].file_writer.add_summary(experiment_tag)
        self._trial_writer[trial].file_writer.add_summary(session_start_tag)
        self._trial_writer[trial].file_writer.add_summary(session_end_tag)
    except Exception:
        logger.exception('TensorboardX failed to log hparams. This may be due to an unsupported type in the hyperparameter values.')