import abc
import logging
import os
from typing import List, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.recipes import dag_help_strings
from mlflow.recipes.artifacts import Artifact
from mlflow.recipes.step import BaseStep, StepClass, StepStatus
from mlflow.recipes.utils import (
from mlflow.recipes.utils.execution import (
from mlflow.recipes.utils.step import display_html
from mlflow.utils.class_utils import _get_class_from_string
def _get_subgraph_for_target_step(self, target_step: BaseStep) -> List[BaseStep]:
    """
        Return a list of step objects representing a connected DAG containing the target_step.
        The returned list should be a sublist of self._steps.
        """
    subgraph = []
    if target_step.step_class == StepClass.UNKNOWN:
        return subgraph
    for step in self._steps:
        if target_step.step_class() == step.step_class():
            subgraph.append(step)
    return subgraph