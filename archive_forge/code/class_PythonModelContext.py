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
class PythonModelContext:
    """
    A collection of artifacts that a :class:`~PythonModel` can use when performing inference.
    :class:`~PythonModelContext` objects are created *implicitly* by the
    :func:`save_model() <mlflow.pyfunc.save_model>` and
    :func:`log_model() <mlflow.pyfunc.log_model>` persistence methods, using the contents specified
    by the ``artifacts`` parameter of these methods.
    """

    def __init__(self, artifacts, model_config):
        """
        Args:
            artifacts: A dictionary of ``<name, artifact_path>`` entries, where ``artifact_path``
                is an absolute filesystem path to a given artifact.
            model_config: The model configuration to make available to the model at
                loading time.
        """
        self._artifacts = artifacts
        self._model_config = model_config

    @property
    def artifacts(self):
        """
        A dictionary containing ``<name, artifact_path>`` entries, where ``artifact_path`` is an
        absolute filesystem path to the artifact.
        """
        return self._artifacts

    @experimental
    @property
    def model_config(self):
        """
        A dictionary containing ``<config, value>`` entries, where ``config`` is the name
        of the model configuration keys and ``value`` is the value of the given configuration.
        """
        return self._model_config