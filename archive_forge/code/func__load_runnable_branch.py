import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def _load_runnable_branch(file_path: Union[Path, str]):
    """Load the model

    Args:
        file_path: Path to file to load the model from.
    """
    from langchain.schema.runnable import RunnableBranch
    load_path = Path(file_path)
    if not load_path.exists() or not load_path.is_dir():
        raise MlflowException(f'File {load_path} must exist and must be a directory in order to load runnable with steps.')
    branches_conf_file = load_path / _RUNNABLE_BRANCHES_FILE_NAME
    if not branches_conf_file.exists():
        raise MlflowException(f'File {branches_conf_file} must exist in order to load runnable with steps.')
    branches_conf = _load_from_yaml(branches_conf_file)
    branches_path = load_path / _BRANCHES_FOLDER_NAME
    if not branches_path.exists() or not branches_path.is_dir():
        raise MlflowException(f'Folder {branches_path} must exist and must be a directory in order to load runnable with steps.')
    branches = []
    for branch in os.listdir(branches_path):
        if branch == _DEFAULT_BRANCH_NAME:
            default_branch_path = branches_path / _DEFAULT_BRANCH_NAME
            default = _load_model_from_path(default_branch_path, branches_conf.get(_DEFAULT_BRANCH_NAME))
        else:
            branch_tuple = []
            for i in range(2):
                config = branches_conf.get(f'{branch}-{i}')
                runnable = _load_model_from_path(os.path.join(branches_path, branch, str(i)), config)
                branch_tuple.append(runnable)
            branches.append(tuple(branch_tuple))
    branches.append(default)
    return RunnableBranch(*branches)