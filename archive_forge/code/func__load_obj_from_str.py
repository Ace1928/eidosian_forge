import functools
import importlib
import sys
import types
import torch
from .allowed_functions import _disallowed_function_ids, is_user_defined_allowed
from .utils import hashable
from .variables import (
def _load_obj_from_str(fully_qualified_name):
    module, obj_name = fully_qualified_name.rsplit('.', maxsplit=1)
    return getattr(importlib.import_module(module), obj_name)