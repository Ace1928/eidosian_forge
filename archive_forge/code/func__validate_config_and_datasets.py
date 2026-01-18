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
def _validate_config_and_datasets(self) -> None:
    if TRAIN_DATASET_KEY not in self.datasets:
        raise KeyError(f"'{TRAIN_DATASET_KEY}' key must be preset in `datasets`. Got {list(self.datasets.keys())}")
    if self.dmatrix_params:
        for key in self.dmatrix_params:
            if key not in self.datasets:
                raise KeyError(f"`dmatrix_params` dict contains key '{key}' which is not present in `datasets`.")