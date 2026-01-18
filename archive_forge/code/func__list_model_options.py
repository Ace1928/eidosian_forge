import importlib
import os
import re
import warnings
from collections import OrderedDict
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import CONFIG_NAME, logging
def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and (not use_model_types):
        raise ValueError('Using `use_model_types=False` requires a `config_to_class` dictionary.')
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {model_type: f'[`{config}`]' for model_type, config in CONFIG_MAPPING_NAMES.items()}
        else:
            model_type_to_name = {model_type: _get_class_name(model_class) for model_type, model_class in config_to_class.items() if model_type in MODEL_NAMES_MAPPING}
        lines = [f'{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)' for model_type in sorted(model_type_to_name.keys())]
    else:
        config_to_name = {CONFIG_MAPPING_NAMES[config]: _get_class_name(clas) for config, clas in config_to_class.items() if config in CONFIG_MAPPING_NAMES}
        config_to_model_name = {config: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING_NAMES.items()}
        lines = [f'{indent}- [`{config_name}`] configuration class: {config_to_name[config_name]} ({config_to_model_name[config_name]} model)' for config_name in sorted(config_to_name.keys())]
    return '\n'.join(lines)