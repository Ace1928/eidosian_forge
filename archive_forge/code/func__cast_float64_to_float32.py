import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
import mlflow.tracking
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _cast_float64_to_float32(self, feeds):
    for input_name, input_type in self.inputs:
        if input_type == 'tensor(float)':
            feed = feeds.get(input_name)
            if feed is not None and feed.dtype == np.float64:
                feeds[input_name] = feed.astype(np.float32)
    return feeds