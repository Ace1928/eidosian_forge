import sys
import types
import typing
import warnings
import simdjson as json
from enum import Enum
from dataclasses import is_dataclass
from .utils import issubclass_safe
def _replace_to_dict(cls, to_dict='to_dict'):
    to_dict_src = _to_dict_source(cls)
    to_dict_module = compile(to_dict_src, '<lazyclass_generated_code>', 'exec')
    to_dict_code = [const for const in to_dict_module.co_consts if isinstance(const, types.CodeType)][0]
    to_dict_func = types.FunctionType(to_dict_code, sys.modules[cls.__module__].__dict__, to_dict)
    setattr(cls, to_dict, to_dict_func)