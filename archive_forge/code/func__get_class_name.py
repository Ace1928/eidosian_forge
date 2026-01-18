import importlib
import os
import re
import warnings
from collections import OrderedDict
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import CONFIG_NAME, logging
def _get_class_name(model_class: Union[str, List[str]]):
    if isinstance(model_class, (list, tuple)):
        return ' or '.join([f'[`{c}`]' for c in model_class if c is not None])
    return f'[`{model_class}`]'