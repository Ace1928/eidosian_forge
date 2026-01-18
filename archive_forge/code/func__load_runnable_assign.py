import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def _load_runnable_assign(file_path: Union[Path, str]):
    """Load the model

    Args:
        file_path: Path to file to load the model from.
    """
    from langchain.schema.runnable.passthrough import RunnableAssign
    load_path = Path(file_path)
    if not load_path.exists() or not load_path.is_dir():
        raise MlflowException(f'File {load_path} must exist and must be a directory in order to load runnable.')
    mapper_file = load_path / _MAPPER_FOLDER_NAME
    if not mapper_file.exists() or not mapper_file.is_dir():
        raise MlflowException(f'Folder {mapper_file} must exist and must be a directory in order to load runnable assign with mapper.')
    mapper = _load_runnable_with_steps(mapper_file, 'RunnableParallel')
    return RunnableAssign(mapper)