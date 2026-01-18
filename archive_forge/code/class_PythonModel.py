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
class PythonModel:
    """
    Represents a generic Python model that evaluates inputs and produces API-compatible outputs.
    By subclassing :class:`~PythonModel`, users can create customized MLflow models with the
    "python_function" ("pyfunc") flavor, leveraging custom inference logic and artifact
    dependencies.
    """
    __metaclass__ = ABCMeta

    def load_context(self, context):
        """
        Loads artifacts from the specified :class:`~PythonModelContext` that can be used by
        :func:`~PythonModel.predict` when evaluating inputs. When loading an MLflow model with
        :func:`~load_model`, this method is called as soon as the :class:`~PythonModel` is
        constructed.

        The same :class:`~PythonModelContext` will also be available during calls to
        :func:`~PythonModel.predict`, but it may be more efficient to override this method
        and load artifacts from the context at model load time.

        Args:
            context: A :class:`~PythonModelContext` instance containing artifacts that the model
                     can use to perform inference.
        """

    def _get_type_hints(self):
        return _extract_type_hints(self.predict, input_arg_index=1)

    @abstractmethod
    def predict(self, context, model_input, params: Optional[Dict[str, Any]]=None):
        """
        Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.
        For more information about the pyfunc input/output API, see the :ref:`pyfunc-inference-api`.

        Args:
            context: A :class:`~PythonModelContext` instance containing artifacts that the model
                     can use to perform inference.
            model_input: A pyfunc-compatible input for the model to evaluate.
            params: Additional parameters to pass to the model for inference.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                 release without warning.
        """