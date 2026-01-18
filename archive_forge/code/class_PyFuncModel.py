import collections
import functools
import importlib
import inspect
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, Tuple, Union
import numpy as np
import pandas
import yaml
import mlflow
import mlflow.pyfunc.loaders
import mlflow.pyfunc.model
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import _DATABRICKS_FS_LOADER_MODULE, MLMODEL_FILE_NAME
from mlflow.models.signature import (
from mlflow.models.utils import (
from mlflow.protos.databricks_pb2 import (
from mlflow.pyfunc.model import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import (
from mlflow.utils import (
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils._spark_utils import modified_environ
from mlflow.utils.annotations import deprecated, developer_stable, experimental
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.requirements_utils import (
class PyFuncModel:
    """
    MLflow 'python function' model.

    Wrapper around model implementation and metadata. This class is not meant to be constructed
    directly. Instead, instances of this class are constructed and returned from
    :py:func:`load_model() <mlflow.pyfunc.load_model>`.

    ``model_impl`` can be any Python object that implements the `Pyfunc interface
    <https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-inference-api>`_, and is
    returned by invoking the model's ``loader_module``.

    ``model_meta`` contains model metadata loaded from the MLmodel file.
    """

    def __init__(self, model_meta: Model, model_impl: Any, predict_fn: str='predict', predict_stream_fn: Optional[str]=None):
        if not hasattr(model_impl, predict_fn):
            raise MlflowException(f'Model implementation is missing required {predict_fn} method.')
        if not model_meta:
            raise MlflowException('Model is missing metadata.')
        self._model_meta = model_meta
        self.__model_impl = model_impl
        self._predict_fn = getattr(model_impl, predict_fn)
        if predict_stream_fn:
            if not hasattr(model_impl, predict_stream_fn):
                raise MlflowException(f'Model implementation is missing required {predict_stream_fn} method.')
            self._predict_stream_fn = getattr(model_impl, predict_stream_fn)
        else:
            self._predict_stream_fn = None

    @property
    @developer_stable
    def _model_impl(self) -> Any:
        """
        The underlying model implementation object.

        NOTE: This is a stable developer API.
        """
        return self.__model_impl

    def _validate_prediction_input(self, data: PyFuncInput, params: Optional[Dict[str, Any]]=None) -> PyFuncInput:
        input_schema = self.metadata.get_input_schema()
        flavor = self.loader_module
        if input_schema is not None:
            try:
                data = _enforce_schema(data, input_schema, flavor)
            except Exception as e:
                raise MlflowException.invalid_parameter_value(f"Failed to enforce schema of data '{data}' with schema '{input_schema}'. Error: {e}")
        params = _validate_params(params, self.metadata)
        _log_warning_if_params_not_in_predict_signature(_logger, params)
        if HAS_PYSPARK and isinstance(data, SparkDataFrame):
            _logger.warning('Input data is a Spark DataFrame. Note that behaviour for Spark DataFrames is model dependent.')
        return (data, params)

    def predict(self, data: PyFuncInput, params: Optional[Dict[str, Any]]=None) -> PyFuncOutput:
        """
        Generates model predictions.

        If the model contains signature, enforce the input schema first before calling the model
        implementation with the sanitized input. If the pyfunc model does not include model schema,
        the input is passed to the model implementation as is. See `Model Signature Enforcement
        <https://www.mlflow.org/docs/latest/models.html#signature-enforcement>`_ for more details.

        Args:
            data: Model input as one of pandas.DataFrame, numpy.ndarray,
                scipy.sparse.(csc_matrix | csr_matrix), List[Any], or
                Dict[str, numpy.ndarray].
                For model signatures with tensor spec inputs
                (e.g. the Tensorflow core / Keras model), the input data type must be one of
                `numpy.ndarray`, `List[numpy.ndarray]`, `Dict[str, numpy.ndarray]` or
                `pandas.DataFrame`. If data is of `pandas.DataFrame` type and the model
                contains a signature with tensor spec inputs, the corresponding column values
                in the pandas DataFrame will be reshaped to the required shape with 'C' order
                (i.e. read / write the elements using C-like index order), and DataFrame
                column values will be cast as the required tensor spec type. For Pyspark
                DataFrame inputs, MLflow will only enforce the schema on a subset
                of the data rows.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions as one of pandas.DataFrame, pandas.Series, numpy.ndarray or list.
        """
        data, params = self._validate_prediction_input(data, params)
        if inspect.signature(self._predict_fn).parameters.get('params'):
            return self._predict_fn(data, params=params)
        return self._predict_fn(data)

    def predict_stream(self, data: PyFuncInput, params: Optional[Dict[str, Any]]=None) -> Iterator[PyFuncOutput]:
        if self._predict_stream_fn is None:
            raise MlflowException('This model does not support predict_stream method.')
        data, params = self._validate_prediction_input(data, params)
        if inspect.signature(self._predict_fn).parameters.get('params'):
            return self._predict_stream_fn(data, params=params)
        return self._predict_stream_fn(data)

    @experimental
    def unwrap_python_model(self):
        """
        Unwrap the underlying Python model object.

        This method is useful for accessing custom model functions, while still being able to
        leverage the MLflow designed workflow through the `predict()` method.

        Returns:
            The underlying wrapped model object

        .. code-block:: python
            :test:
            :caption: Example

            import mlflow


            # define a custom model
            class MyModel(mlflow.pyfunc.PythonModel):
                def predict(self, context, model_input, params=None):
                    return self.my_custom_function(model_input, params)

                def my_custom_function(self, model_input, params=None):
                    # do something with the model input
                    return 0


            some_input = 1
            # save the model
            with mlflow.start_run():
                model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=MyModel())

            # load the model
            loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
            print(type(loaded_model))  # <class 'mlflow.pyfunc.model.PyFuncModel'>
            unwrapped_model = loaded_model.unwrap_python_model()
            print(type(unwrapped_model))  # <class '__main__.MyModel'>

            # does not work, only predict() is exposed
            # print(loaded_model.my_custom_function(some_input))
            print(unwrapped_model.my_custom_function(some_input))  # works
            print(loaded_model.predict(some_input))  # works

            # works, but None is needed for context arg
            print(unwrapped_model.predict(None, some_input))
        """
        try:
            python_model = self._model_impl.python_model
            if python_model is None:
                raise AttributeError('Expected python_model attribute not to be None.')
        except AttributeError as e:
            raise MlflowException('Unable to retrieve base model object from pyfunc.') from e
        return python_model

    def __eq__(self, other):
        if not isinstance(other, PyFuncModel):
            return False
        return self._model_meta == other._model_meta

    @property
    def metadata(self):
        """Model metadata."""
        if self._model_meta is None:
            raise MlflowException('Model is missing metadata.')
        return self._model_meta

    @experimental
    @property
    def model_config(self):
        """Model's flavor configuration"""
        return self._model_meta.flavors[FLAVOR_NAME].get(MODEL_CONFIG, {})

    @experimental
    @property
    def loader_module(self):
        """Model's flavor configuration"""
        if self._model_meta.flavors.get(FLAVOR_NAME) is None:
            return None
        return self._model_meta.flavors[FLAVOR_NAME].get(MAIN)

    def __repr__(self):
        info = {}
        if self._model_meta is not None:
            if hasattr(self._model_meta, 'run_id') and self._model_meta.run_id is not None:
                info['run_id'] = self._model_meta.run_id
            if hasattr(self._model_meta, 'artifact_path') and self._model_meta.artifact_path is not None:
                info['artifact_path'] = self._model_meta.artifact_path
            info['flavor'] = self._model_meta.flavors[FLAVOR_NAME]['loader_module']
        return yaml.safe_dump({'mlflow.pyfunc.loaded_model': info}, default_flow_style=False)