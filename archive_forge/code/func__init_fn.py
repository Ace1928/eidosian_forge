import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import itertools
import abc
import _thread
from types import FunctionType, GenericAlias
def _init_fn(fields, std_fields, kw_only_fields, frozen, has_post_init, self_name, globals, slots):
    seen_default = False
    for f in std_fields:
        if f.init:
            if not (f.default is MISSING and f.default_factory is MISSING):
                seen_default = True
            elif seen_default:
                raise TypeError(f'non-default argument {f.name!r} follows default argument')
    locals = {f'_type_{f.name}': f.type for f in fields}
    locals.update({'MISSING': MISSING, '_HAS_DEFAULT_FACTORY': _HAS_DEFAULT_FACTORY, '__dataclass_builtins_object__': object})
    body_lines = []
    for f in fields:
        line = _field_init(f, frozen, locals, self_name, slots)
        if line:
            body_lines.append(line)
    if has_post_init:
        params_str = ','.join((f.name for f in fields if f._field_type is _FIELD_INITVAR))
        body_lines.append(f'{self_name}.{_POST_INIT_NAME}({params_str})')
    if not body_lines:
        body_lines = ['pass']
    _init_params = [_init_param(f) for f in std_fields]
    if kw_only_fields:
        _init_params += ['*']
        _init_params += [_init_param(f) for f in kw_only_fields]
    return _create_fn('__init__', [self_name] + _init_params, body_lines, locals=locals, globals=globals, return_type=None)