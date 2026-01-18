import os
from typing import Any, Dict, Union
import lightgbm
import lightgbm_ray
import xgboost_ray
from lightgbm_ray.tune import TuneReportCheckpointCallback
from ray.train import Checkpoint
from ray.train.gbdt_trainer import GBDTTrainer
from ray.train.lightgbm import LightGBMCheckpoint
from ray.util.annotations import PublicAPI
def _model_iteration(self, model: Union[lightgbm.LGBMModel, lightgbm.Booster]) -> int:
    if isinstance(model, lightgbm.Booster):
        return model.current_iteration()
    return model.booster_.current_iteration()