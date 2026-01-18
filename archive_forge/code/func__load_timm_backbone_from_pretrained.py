import copy
import importlib
import json
import os
import warnings
from collections import OrderedDict
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import (
from .configuration_auto import AutoConfig, model_type_to_module_name, replace_list_option_in_docstrings
@classmethod
def _load_timm_backbone_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    requires_backends(cls, ['vision', 'timm'])
    from ...models.timm_backbone import TimmBackboneConfig
    config = kwargs.pop('config', TimmBackboneConfig())
    if kwargs.get('out_features', None) is not None:
        raise ValueError('Cannot specify `out_features` for timm backbones')
    if kwargs.get('output_loading_info', False):
        raise ValueError('Cannot specify `output_loading_info=True` when loading from timm')
    num_channels = kwargs.pop('num_channels', config.num_channels)
    features_only = kwargs.pop('features_only', config.features_only)
    use_pretrained_backbone = kwargs.pop('use_pretrained_backbone', config.use_pretrained_backbone)
    out_indices = kwargs.pop('out_indices', config.out_indices)
    config = TimmBackboneConfig(backbone=pretrained_model_name_or_path, num_channels=num_channels, features_only=features_only, use_pretrained_backbone=use_pretrained_backbone, out_indices=out_indices)
    return super().from_config(config, **kwargs)