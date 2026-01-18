import logging
from pathlib import Path
from typing import Any, Dict
import mlflow
from mlflow.entities import SourceType
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.recipes.artifacts import ModelVersionArtifact, RegisteredModelVersionInfo
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.steps.train import TrainStep
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.tracking import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.databricks_utils import (
from mlflow.utils.mlflow_tags import MLFLOW_RECIPE_TEMPLATE_NAME, MLFLOW_SOURCE_TYPE
def _build_card(self, run_id: str) -> BaseCard:
    card = BaseCard(self.recipe_name, self.name)
    card_tab = card.add_tab('Run Summary', '{{ MODEL_NAME }}' + '{{ MODEL_VERSION }}' + '{{ MODEL_SOURCE_URI }}' + '{{ ALERTS }}' + '{{ EXE_DURATION }}' + '{{ LAST_UPDATE_TIME }}')
    if self.version is not None:
        model_version_url = get_databricks_model_version_url(registry_uri=mlflow.get_registry_uri(), name=self.register_model_name, version=self.version)
        if model_version_url is not None:
            card_tab.add_html('MODEL_NAME', f'<b>Model Name:</b> <a href={model_version_url}>{self.register_model_name}</a><br><br>')
            card_tab.add_html('MODEL_VERSION', f'<b>Model Version</b> <a href={model_version_url}>{self.version}</a><br><br>')
        else:
            card_tab.add_markdown('MODEL_NAME', f'**Model Name:** `{self.register_model_name}`')
            card_tab.add_markdown('MODEL_VERSION', f'**Model Version:** `{self.version}`')
    model_source_url = get_databricks_run_url(tracking_uri=mlflow.get_tracking_uri(), run_id=run_id, artifact_path=f'train/{TrainStep.MODEL_ARTIFACT_RELATIVE_PATH}')
    if self.model_uri is not None and model_source_url is not None:
        card_tab.add_html('MODEL_SOURCE_URI', f'<b>Model Source URI</b> <a href={model_source_url}>{self.model_uri}</a>')
    elif self.model_uri is not None:
        card_tab.add_markdown('MODEL_SOURCE_URI', f'**Model Source URI:** `{self.model_uri}`')
    return card