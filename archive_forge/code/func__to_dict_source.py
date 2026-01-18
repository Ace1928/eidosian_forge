import sys
import types
import typing
import warnings
import simdjson as json
from enum import Enum
from dataclasses import is_dataclass
from .utils import issubclass_safe
def _to_dict_source(cls):
    lines = ['def to_dict(self):', '    result = {}']
    INCLUDE_NONES = False
    for name, field_type in typing.get_type_hints(cls).items():
        access = f'self.{name}'
        transform = expr_builder_to(field_type)
        if transform('x') != 'x':
            if typing.get_origin(field_type) == typing.Union:
                transform = expr_builder_to(typing.get_args(field_type)[0])
            lines.append(f'    value = {access}')
            lines.append(f'    if value is not None:')
            lines.append(f'        value = ' + transform('value'))
            if INCLUDE_NONES:
                lines.append(f'    result[{name!r}] = value')
            else:
                lines.append(f'        result[{name!r}] = value')
        else:
            lines.append(f'    result[{name!r}] = {access}')
    lines.append('    return result')
    lines.append('')
    return '\n'.join(lines)