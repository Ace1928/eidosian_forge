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
@classmethod
def _validate_scaling_config(cls, scaling_config: ScalingConfig) -> ScalingConfig:
    if scaling_config.trainer_resources not in [None, {}]:
        raise ValueError(f'The `trainer_resources` attribute for {cls.__name__} is currently ignored and defaults to `{{}}`. Remove the `trainer_resources` key from your `ScalingConfig` to resolve.')
    return super(GBDTTrainer, cls)._validate_scaling_config(scaling_config=scaling_config)