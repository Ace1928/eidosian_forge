import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def _load_runnable_with_steps(file_path: Union[Path, str], model_type: str):
    """Load the model

    Args:
        file_path: Path to file to load the model from.
        model_type: Type of the model to load.
    """
    from langchain.schema.runnable import RunnableParallel, RunnableSequence
    load_path = Path(file_path)
    if not load_path.exists() or not load_path.is_dir():
        raise MlflowException(f'File {load_path} must exist and must be a directory in order to load runnable with steps.')
    steps_conf_file = load_path / _RUNNABLE_STEPS_FILE_NAME
    if not steps_conf_file.exists():
        raise MlflowException(f'File {steps_conf_file} must exist in order to load runnable with steps.')
    steps_conf = _load_from_yaml(steps_conf_file)
    steps_path = load_path / _STEPS_FOLDER_NAME
    if not steps_path.exists() or not steps_path.is_dir():
        raise MlflowException(f'Folder {steps_path} must exist and must be a directory in order to load runnable with steps.')
    steps = {}
    for step in (f for f in os.listdir(steps_path) if not f.startswith('.')):
        config = steps_conf.get(step)
        runnable = _load_model_from_path(os.path.join(steps_path, step), config)
        steps[step] = runnable
    if model_type == RunnableSequence.__name__:
        steps = [value for _, value in sorted(steps.items(), key=lambda item: int(item[0]))]
        return runnable_sequence_from_steps(steps)
    if model_type == RunnableParallel.__name__:
        return RunnableParallel(steps)