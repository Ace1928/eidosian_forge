import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import yaml
import mlflow.tracking
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _run_and_log_function(self, original, args, kwargs, unlogged_params, is_fine_tune=False):
    mlflow_cbs = [cb for cb in self.cbs if cb.name == '___mlflow_fastai']
    fit_in_fine_tune = original.__name__ == 'fit' and len(mlflow_cbs) > 0 and mlflow_cbs[0].is_fine_tune
    if not fit_in_fine_tune:
        log_fn_args_as_params(original, list(args), kwargs, unlogged_params)
    run_id = mlflow.active_run().info.run_id
    with batch_metrics_logger(run_id) as metrics_logger:
        if not fit_in_fine_tune:
            early_stop_callback = _find_callback_of_type(EarlyStoppingCallback, self.cbs)
            _log_early_stop_callback_params(early_stop_callback)
            self.remove_cbs(mlflow_cbs)
            with self.no_bar(), self.no_logging():
                _log_model_info(learner=self)
            mlflowFastaiCallback = getFastaiCallback(metrics_logger=metrics_logger, is_fine_tune=is_fine_tune)
            self.add_cb(mlflowFastaiCallback)
        return original(self, *args, **kwargs)