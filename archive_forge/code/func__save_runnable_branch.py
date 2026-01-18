import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def _save_runnable_branch(model, file_path, loader_fn, persist_dir):
    """
    Save runnable branch in to path.
    """
    save_path = Path(file_path)
    save_path.mkdir(parents=True, exist_ok=True)
    branches_path = save_path / _BRANCHES_FOLDER_NAME
    branches_path.mkdir()
    unsaved_runnables = {}
    branches_conf = {}
    for index, branch_tuple in enumerate(model.branches):
        for i, runnable in enumerate(branch_tuple):
            save_runnable_path = branches_path / str(index) / str(i)
            save_runnable_path.mkdir(parents=True)
            branches_conf[f'{index}-{i}'] = {}
            try:
                branches_conf[f'{index}-{i}'] = _save_internal_runnables(runnable, save_runnable_path, loader_fn, persist_dir)
            except Exception as e:
                unsaved_runnables[f'{index}-{i}'] = f'{runnable} -- {e}'
    default_branch_path = branches_path / _DEFAULT_BRANCH_NAME
    default_branch_path.mkdir()
    try:
        branches_conf[_DEFAULT_BRANCH_NAME] = _save_internal_runnables(model.default, default_branch_path, loader_fn, persist_dir)
    except Exception as e:
        unsaved_runnables[_DEFAULT_BRANCH_NAME] = f'{model.default} -- {e}'
    if unsaved_runnables:
        raise MlflowException(f'Failed to save runnable branch: {unsaved_runnables}.')
    with save_path.joinpath(_RUNNABLE_BRANCHES_FILE_NAME).open('w') as f:
        yaml.dump(branches_conf, f, default_flow_style=False)