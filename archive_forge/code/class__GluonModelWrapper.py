import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import deprecated
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
class _GluonModelWrapper:

    def __init__(self, gluon_model):
        self.gluon_model = gluon_model

    def predict(self, data, params: Optional[Dict[str, Any]]=None):
        """This is a docstring. Here is more info.

        Args:
            data: Either a pandas DataFrame or a numpy array containing input array values.
                If the input is a DataFrame, it will be converted to an array first by a
                `ndarray = df.values`.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions. If the input is a pandas.DataFrame, the predictions are returned
            in a pandas.DataFrame. If the input is a numpy array, the predictions are returned
            as either a numpy.ndarray or a plain list for hybrid models.

        """
        import mxnet as mx
        if isinstance(data, pd.DataFrame):
            ndarray = mx.nd.array(data.values)
            preds = self.gluon_model(ndarray)
            if isinstance(preds, mx.ndarray.ndarray.NDArray):
                preds = preds.asnumpy()
            return pd.DataFrame(preds)
        elif isinstance(data, np.ndarray):
            if Version(mx.__version__) >= Version('2.0.0'):
                ndarray = mx.np.array(data)
            else:
                ndarray = mx.nd.array(data)
            preds = self.gluon_model(ndarray)
            if isinstance(preds, mx.ndarray.ndarray.NDArray):
                preds = preds.asnumpy()
            return preds
        else:
            raise TypeError('Input data should be pandas.DataFrame or numpy.ndarray')