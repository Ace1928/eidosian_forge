import sys
import types
import typing
import warnings
import simdjson as json
from enum import Enum
from dataclasses import is_dataclass
from .utils import issubclass_safe
def _replace_from_dict(cls, from_dict='from_dict'):
    from_dict_src = _from_dict_source(cls)
    from_dict_module = compile(from_dict_src, '<lazyclass_generated_code>', 'exec')
    from_dict_code = [const for const in from_dict_module.co_consts if isinstance(const, types.CodeType)][0]
    the_globals = {**sys.modules[cls.__module__].__dict__, **referenced_types(cls)}
    from_dict_func = types.FunctionType(from_dict_code, the_globals, from_dict)
    setattr(cls, from_dict, classmethod(from_dict_func))