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
def _repartition_datasets_to_match_num_actors(self):
    for dataset_key, dataset in self.datasets.items():
        if dataset.num_blocks() < self._ray_params.num_actors:
            if dataset.size_bytes() > _WARN_REPARTITION_THRESHOLD:
                warnings.warn(f"Dataset '{dataset_key}' has {dataset.num_blocks()} blocks, which is less than the `num_workers` {self._ray_params.num_actors}. This dataset will be automatically repartitioned to {self._ray_params.num_actors} blocks. You can disable this error message by partitioning the dataset to have blocks >= number of workers via `dataset.repartition(num_workers)`.")
            self.datasets[dataset_key] = dataset.repartition(self._ray_params.num_actors)