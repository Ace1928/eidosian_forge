import logging
import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Type
from ray import train, tune
from ray._private.dict import flatten_dict
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.constants import MODEL_KEY, TRAIN_DATASET_KEY
from ray.train.trainer import BaseTrainer, GenDataset
from ray.tune import Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.util.annotations import DeveloperAPI
def _checkpoint_at_end(self, model, evals_result: dict) -> None:
    result_dict = flatten_dict(evals_result, delimiter='-')
    for k in list(result_dict):
        result_dict[k] = result_dict[k][-1]
    if getattr(self._tune_callback_checkpoint_cls, '_report_callbacks_cls', None):
        with tune.checkpoint_dir(step=self._model_iteration(model)) as cp_dir:
            self._save_model(model, path=os.path.join(cp_dir, MODEL_KEY))
        tune.report(**result_dict)
    else:
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            self._save_model(model, path=checkpoint_dir)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(result_dict, checkpoint=checkpoint)