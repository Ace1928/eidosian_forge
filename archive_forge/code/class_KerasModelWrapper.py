import os
import keras
import numpy as np
import pandas as pd
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
class KerasModelWrapper:

    def __init__(self, model, signature, save_exported_model=False):
        self.model = model
        self.signature = signature
        self.save_exported_model = save_exported_model

    def get_model_call_method(self):
        if self.save_exported_model:
            return self.model.serve
        else:
            return self.model.predict

    def predict(self, data, **kwargs):
        model_call = self.get_model_call_method()
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(model_call(data.values), index=data.index)
        supported_input_types = (np.ndarray, list, tuple, dict)
        if not isinstance(data, supported_input_types):
            raise MlflowException(f'`data` must be one of: {[x.__name__ for x in supported_input_types]}, but received type: {type(data)}.', INVALID_PARAMETER_VALUE)
        return model_call(data)