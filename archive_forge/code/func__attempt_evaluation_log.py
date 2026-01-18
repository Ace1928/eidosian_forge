import logging
import operator
import os
import shutil
import sys
from itertools import chain
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # noqa: N812
import wandb
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from wandb.sdk.lib.deprecate import Deprecated, deprecate
from wandb.util import add_import_hook
def _attempt_evaluation_log(self, commit=True):
    if self.log_evaluation and self._validation_data_logger:
        try:
            if not self.model:
                wandb.termwarn('WandbCallback unable to read model from trainer')
            else:
                self._validation_data_logger.log_predictions(predictions=self._validation_data_logger.make_predictions(self.model.predict), commit=commit)
                self._model_trained_since_last_eval = False
        except Exception as e:
            wandb.termwarn('Error durring prediction logging for epoch: ' + str(e))