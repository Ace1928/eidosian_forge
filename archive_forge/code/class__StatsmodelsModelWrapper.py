import inspect
import itertools
import logging
import os
from typing import Any, Dict, Optional
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.validation import _is_numeric
class _StatsmodelsModelWrapper:

    def __init__(self, statsmodels_model):
        self.statsmodels_model = statsmodels_model

    def predict(self, dataframe, params: Optional[Dict[str, Any]]=None):
        """
        Args:
            dataframe: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions.
        """
        from statsmodels.tsa.base.tsa_model import TimeSeriesModel
        model = self.statsmodels_model.model
        if isinstance(model, TimeSeriesModel):
            if dataframe.shape[0] != 1 or not ('start' in dataframe.columns and 'end' in dataframe.columns):
                raise MlflowException('prediction dataframes for a TimeSeriesModel must have exactly one row' + ' and include columns called start and end')
            start_date = dataframe['start'][0]
            end_date = dataframe['end'][0]
            return self.statsmodels_model.predict(start=start_date, end=end_date)
        else:
            return self.statsmodels_model.predict(dataframe)