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
def _convert_scaling_config_to_ray_params(scaling_config: ScalingConfig, ray_params_cls: Type['xgboost_ray.RayParams'], default_ray_params: Optional[Dict[str, Any]]=None) -> 'xgboost_ray.RayParams':
    """Scaling config parameters have precedence over default ray params.

    Default ray params are defined in the trainers (xgboost/lightgbm),
    but if the user requests something else, that should be respected.
    """
    resources = (scaling_config.resources_per_worker or {}).copy()
    cpus_per_actor = resources.pop('CPU', 0)
    if not cpus_per_actor:
        cpus_per_actor = default_ray_params.get('cpus_per_actor', 0)
    gpus_per_actor = resources.pop('GPU', int(scaling_config.use_gpu))
    if not gpus_per_actor:
        gpus_per_actor = default_ray_params.get('gpus_per_actor', 0)
    resources_per_actor = resources
    if not resources_per_actor:
        resources_per_actor = default_ray_params.get('resources_per_actor', None)
    num_actors = scaling_config.num_workers
    if not num_actors:
        num_actors = default_ray_params.get('num_actors', 0)
    ray_params_kwargs = default_ray_params.copy() or {}
    ray_params_kwargs.update({'cpus_per_actor': int(cpus_per_actor), 'gpus_per_actor': int(gpus_per_actor), 'resources_per_actor': resources_per_actor, 'num_actors': int(num_actors)})
    if not hasattr(ray_params_cls, 'placement_options'):

        @dataclass
        class RayParamsFromScalingConfig(ray_params_cls):
            placement_options: Dict[str, Any] = None

            def get_tune_resources(self) -> PlacementGroupFactory:
                pgf = super().get_tune_resources()
                placement_options = self.placement_options.copy()
                extended_pgf = PlacementGroupFactory(pgf.bundles, **placement_options)
                extended_pgf._head_bundle_is_empty = pgf._head_bundle_is_empty
                return extended_pgf
        ray_params_cls_extended = RayParamsFromScalingConfig
    else:
        ray_params_cls_extended = ray_params_cls
    placement_options = {'strategy': scaling_config.placement_strategy}
    ray_params = ray_params_cls_extended(placement_options=placement_options, **ray_params_kwargs)
    return ray_params