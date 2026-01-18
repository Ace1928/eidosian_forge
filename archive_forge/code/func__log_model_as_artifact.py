import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast
import xgboost as xgb  # type: ignore
from xgboost import Booster
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
def _log_model_as_artifact(self, model: Booster) -> None:
    model_name = f'{wandb.run.id}_model.json'
    model_path = Path(wandb.run.dir) / model_name
    model.save_model(str(model_path))
    model_artifact = wandb.Artifact(name=model_name, type='model')
    model_artifact.add_file(str(model_path))
    wandb.log_artifact(model_artifact)