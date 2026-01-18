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
def _get_execution_directory_basename(recipe_root_path):
    """
    Obtains the basename of the execution directory corresponding to the specified recipe.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.

    Returns:
        The basename of the execution directory corresponding to the specified recipe.
    """
    return hashlib.sha256(os.path.abspath(recipe_root_path).encode('utf-8')).hexdigest()