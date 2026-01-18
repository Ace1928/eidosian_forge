import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def _save_runnable_assign(model, file_path, loader_fn=None, persist_dir=None):
    from langchain.schema.runnable import RunnableParallel
    save_path = Path(file_path)
    save_path.mkdir(parents=True, exist_ok=True)
    mapper_path = save_path / _MAPPER_FOLDER_NAME
    mapper_path.mkdir()
    if not isinstance(model.mapper, RunnableParallel):
        raise MlflowException(f"Failed to save model {model} with type {model.__class__.__name__}. RunnableAssign's mapper must be a RunnableParallel.")
    _save_runnable_with_steps(model.mapper, mapper_path, loader_fn, persist_dir)