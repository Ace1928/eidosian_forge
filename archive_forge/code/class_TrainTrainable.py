import abc
import copy
import inspect
import json
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union
import pyarrow.fs
import ray
import ray.cloudpickle as pickle
from ray._private.dict import merge_dicts
from ray.air._internal import usage as air_usage
from ray.air._internal.config import ensure_only_allowed_dataclass_keys_updated
from ray.air._internal.usage import AirEntrypoint
from ray.air.config import RunConfig, ScalingConfig
from ray.air.result import Result
from ray.train import Checkpoint
from ray.train._internal.session import _get_session
from ray.train._internal.storage import _exists_at_fs_path, get_fs_and_path
from ray.train.constants import TRAIN_DATASET_KEY
from ray.util import PublicAPI
from ray.util.annotations import DeveloperAPI
class TrainTrainable(trainable_cls):
    """Adds default resources to the Trainable."""
    _handles_checkpoint_freq = trainer_cls._handles_checkpoint_freq
    _handles_checkpoint_at_end = trainer_cls._handles_checkpoint_at_end

    @classmethod
    def has_base_dataset(cls) -> bool:
        """Whether a dataset is provided through the Trainer."""
        return has_base_dataset

    @classmethod
    def base_scaling_config(cls) -> ScalingConfig:
        """Returns the unchanged scaling config provided through the Trainer."""
        return scaling_config

    def setup(self, config, **kwargs):
        base_config = dict(kwargs)
        run_config = base_config.pop('run_config', None)
        self._merged_config = merge_dicts(base_config, self.config)
        self._merged_config['run_config'] = run_config
        merged_scaling_config = self._merged_config.get('scaling_config')
        if isinstance(merged_scaling_config, dict):
            merged_scaling_config = ScalingConfig(**merged_scaling_config)
        self._merged_config['scaling_config'] = self._reconcile_scaling_config_with_trial_resources(merged_scaling_config)
        if self.has_base_dataset():
            DataContext._set_current(dataset_context)
        super(TrainTrainable, self).setup(config)

    def _reconcile_scaling_config_with_trial_resources(self, scaling_config: ScalingConfig) -> ScalingConfig:
        """
                ResourceChangingScheduler workaround.

                Ensures that the scaling config matches trial resources.

                This should be replaced with RCS returning a ScalingConfig
                in the future.
                """
        trial_resources = self.trial_resources
        if not isinstance(trial_resources, PlacementGroupFactory):
            return scaling_config
        if scaling_config:
            scaling_config = trainer_cls._validate_scaling_config(scaling_config)
        scaling_config_from_trial_resources = ScalingConfig.from_placement_group_factory(trial_resources)
        if scaling_config_from_trial_resources != scaling_config:
            scaling_config = trainer_cls._validate_scaling_config(scaling_config_from_trial_resources)
        return scaling_config

    def _trainable_func(self, config):
        super()._trainable_func(self._merged_config)

    @classmethod
    def default_resource_request(cls, config):
        updated_scaling_config = config.get('scaling_config', scaling_config)
        if isinstance(updated_scaling_config, dict):
            updated_scaling_config = ScalingConfig(**updated_scaling_config)
        validated_scaling_config = trainer_cls._validate_scaling_config(updated_scaling_config)
        return validated_scaling_config.as_placement_group_factory()