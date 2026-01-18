from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
def _parse_optional_string_set(self, name, value):
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError('%s: expected `None` or collection of strings; got %r: %r' % (name, type(value), value))
    value = frozenset(value)
    for item in value:
        if not isinstance(item, str):
            raise TypeError('%s: expected `None` or collection of strings; got item of type %r: %r' % (name, type(item), item))
    return value