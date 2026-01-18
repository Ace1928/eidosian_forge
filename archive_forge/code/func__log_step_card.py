import abc
import json
import logging
import os
import time
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional
import yaml
from mlflow.recipes.cards import CARD_HTML_NAME, CARD_PICKLE_NAME, BaseCard, FailureCard
from mlflow.recipes.utils import get_recipe_name
from mlflow.recipes.utils.step import display_html
from mlflow.tracking import MlflowClient
from mlflow.utils.databricks_utils import is_in_databricks_runtime
def _log_step_card(self, run_id: str, step_name: str) -> None:
    """
        Logs a step card as an artifact (destination: <step_name>/card.html) in a specified run.
        If the step card does not exist, logging is skipped.

        Args:
            run_id: Run ID to which the step card is logged.
            step_name: Step name.
        """
    from mlflow.recipes.utils.execution import get_step_output_path
    local_card_path = get_step_output_path(recipe_root_path=self.recipe_root, step_name=step_name, relative_path=CARD_HTML_NAME)
    if os.path.exists(local_card_path):
        MlflowClient().log_artifact(run_id, local_card_path, artifact_path=step_name)
    else:
        _logger.warning('Failed to log step card for step %s. Run ID: %s. Card local path: %s', step_name, run_id, local_card_path)