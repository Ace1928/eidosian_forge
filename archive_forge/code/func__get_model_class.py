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
def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models
    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, 'architectures', [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f'TF{arch}' in name_to_model:
            return name_to_model[f'TF{arch}']
        elif f'Flax{arch}' in name_to_model:
            return name_to_model[f'Flax{arch}']
    return supported_models[0]