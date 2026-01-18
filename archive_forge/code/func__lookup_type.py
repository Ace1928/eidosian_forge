from copy import deepcopy
from functools import partial
import importlib
import json
import os
import re
import yaml
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils import force_list, merge_dicts
def _lookup_type(cls, type_):
    if cls is not None and hasattr(cls, '__type_registry__') and isinstance(cls.__type_registry__, dict) and (type_ in cls.__type_registry__ or (isinstance(type_, str) and re.sub('[\\W_]', '', type_.lower()) in cls.__type_registry__)):
        available_class_for_type = cls.__type_registry__.get(type_)
        if available_class_for_type is None:
            available_class_for_type = cls.__type_registry__[re.sub('[\\W_]', '', type_.lower())]
        return available_class_for_type
    return None