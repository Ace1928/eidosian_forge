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
@staticmethod
def _infer_cached_steps(rule_name, steps_to_run, recipe_step_names) -> List[str]:
    """
        Infer cached steps.

        Args:
            rule_name: The name of the Make rule to run.
            steps_to_run: The step names obtained by parsing the Make output showing
                which steps will be executed.
            recipe_step_names: A list of all the step names contained in the specified
                recipe sorted by the execution order.

        """
    index = recipe_step_names.index(rule_name)
    if index == 0:
        return []
    if len(steps_to_run) == 0:
        return recipe_step_names[:index + 1]
    first_step_index = min([recipe_step_names.index(step) for step in steps_to_run])
    return recipe_step_names[:first_step_index]