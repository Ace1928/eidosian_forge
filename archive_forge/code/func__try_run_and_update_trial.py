import copy
import os
import traceback
import warnings
from keras_tuner.src import backend
from keras_tuner.src import config as config_module
from keras_tuner.src import errors
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.distribute import utils as dist_utils
from keras_tuner.src.engine import hypermodel as hm_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner_utils
def _try_run_and_update_trial(self, trial, *fit_args, **fit_kwargs):
    try:
        self._run_and_update_trial(trial, *fit_args, **fit_kwargs)
        trial.status = trial_module.TrialStatus.COMPLETED
        return
    except Exception as e:
        if isinstance(e, errors.FatalError):
            raise e
        if config_module.DEBUG:
            traceback.print_exc()
        if isinstance(e, errors.FailedTrialError):
            trial.status = trial_module.TrialStatus.FAILED
        else:
            trial.status = trial_module.TrialStatus.INVALID
        message = traceback.format_exc()
        trial.message = message