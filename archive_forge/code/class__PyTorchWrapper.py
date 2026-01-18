import atexit
import importlib
import logging
import os
import posixpath
import shutil
import warnings
from functools import partial
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_DEFAULT_PREDICTION_DEVICE
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.checkpoint_utils import download_checkpoint_artifact
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
class _PyTorchWrapper:
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, pytorch_model, device):
        self.pytorch_model = pytorch_model
        self.device = device

    def predict(self, data, params: Optional[Dict[str, Any]]=None):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                            release without warning.

        Returns:
            Model predictions.
        """
        import torch
        if params and 'device' in params:
            raise ValueError("device' can no longer be specified as an inference parameter. It must be specified at load time. Please specify the device at load time, for example: `mlflow.pyfunc.load_model(model_uri, model_config={'device': 'cuda'})`.")
        if isinstance(data, pd.DataFrame):
            inp_data = data.values.astype(np.float32)
        elif isinstance(data, np.ndarray):
            inp_data = data
        elif isinstance(data, (list, dict)):
            raise TypeError('The PyTorch flavor does not support List or Dict input types. Please use a pandas.DataFrame or a numpy.ndarray')
        else:
            raise TypeError('Input data should be pandas.DataFrame or numpy.ndarray')
        device = self.device
        with torch.no_grad():
            input_tensor = torch.from_numpy(inp_data).to(device)
            preds = self.pytorch_model(input_tensor)
            if device != _TORCH_CPU_DEVICE_NAME:
                preds = preds.to(_TORCH_CPU_DEVICE_NAME)
            if not isinstance(preds, torch.Tensor):
                raise TypeError(f"Expected PyTorch model to output a single output tensor, but got output of type '{type(preds)}'")
            if isinstance(data, pd.DataFrame):
                predicted = pd.DataFrame(preds.numpy())
                predicted.index = data.index
            else:
                predicted = preds.numpy()
            return predicted