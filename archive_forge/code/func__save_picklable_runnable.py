import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def _save_picklable_runnable(model, path):
    if not path.endswith('.pkl'):
        raise ValueError(f'File path must end with .pkl, got {path}.')
    with open(path, 'wb') as f:
        cloudpickle.dump(model, f)