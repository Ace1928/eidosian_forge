import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def _load_runnables(path, conf):
    model_type = conf.get(_MODEL_TYPE_KEY)
    model_data = conf.get(_MODEL_DATA_KEY, _MODEL_DATA_YAML_FILE_NAME)
    if model_type in (x.__name__ for x in lc_runnable_with_steps_types()):
        return _load_runnable_with_steps(os.path.join(path, model_data), model_type)
    if model_type in (x.__name__ for x in picklable_runnable_types()) or model_data == _MODEL_DATA_PKL_FILE_NAME:
        return _load_from_pickle(os.path.join(path, model_data))
    if model_type in (x.__name__ for x in lc_runnable_branch_types()):
        return _load_runnable_branch(os.path.join(path, model_data))
    if model_type in (x.__name__ for x in lc_runnable_assign_types()):
        return _load_runnable_assign(os.path.join(path, model_data))
    raise MlflowException.invalid_parameter_value(_UNSUPPORTED_MODEL_ERROR_MESSAGE.format(instance_type=model_type))