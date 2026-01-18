import inspect
import logging
import os
import shutil
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import cloudpickle
import yaml
import mlflow.pyfunc
import mlflow.utils
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _extract_type_hints
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import ChatMessage, ChatParams, ChatResponse
from mlflow.utils.annotations import experimental
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.model_utils import _get_flavor_configuration, _validate_and_copy_code_paths
from mlflow.utils.requirements_utils import _get_pinned_requirement
class _FunctionPythonModel(PythonModel):
    """
    When a user specifies a ``python_model`` argument that is a function, we wrap the function
    in an instance of this class.
    """

    def __init__(self, func, hints=None, signature=None):
        self.func = func
        self.hints = hints
        self.signature = signature

    def _get_type_hints(self):
        return _extract_type_hints(self.func, input_arg_index=0)

    def predict(self, context, model_input, params: Optional[Dict[str, Any]]=None):
        """
        Args:
            context: A instance containing artifacts that the model
                can use to perform inference.
            model_input: A pyfunc-compatible input for the model to evaluate.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions.
        """
        if inspect.signature(self.func).parameters.get('params'):
            return self.func(model_input, params=params)
        _log_warning_if_params_not_in_predict_signature(_logger, params)
        return self.func(model_input)