import hashlib
import logging
import os
import pathlib
import re
import shutil
from typing import Dict, List
from mlflow.environment_variables import (
from mlflow.recipes.step import BaseStep, StepStatus
from mlflow.utils.file_utils import read_yaml, write_yaml
from mlflow.utils.process import _exec_cmd
def _get_step_output_directory_path(execution_directory_path: str, step_name: str) -> str:
    """
    Obtains the path of the local filesystem directory containing outputs for the specified step,
    which may or may not exist.

    Args:
        execution_directory_path: The absolute path of the execution directory on the local
            filesystem for the relevant recipe. The Makefile is created in this directory.
        step_name: The name of the recipe step for which to obtain the output directory path.

    Returns:
        The absolute path of the local filesystem directory containing outputs for the specified
        step.
    """
    return os.path.abspath(os.path.join(execution_directory_path, _STEPS_SUBDIRECTORY_NAME, step_name, _STEP_OUTPUTS_SUBDIRECTORY_NAME))