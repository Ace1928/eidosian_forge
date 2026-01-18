import inspect
import sys
from datetime import datetime, timezone
from collections import Counter
from typing import (Collection, Mapping, Optional, TypeVar, Any, Type, Tuple,
def _handle_undefined_parameters_safe(cls, kvs, usage: str):
    """
    Checks if an undefined parameters action is defined and performs the
    according action.
    """
    undefined_parameter_action = _undefined_parameter_action_safe(cls)
    usage = usage.lower()
    if undefined_parameter_action is None:
        return kvs if usage != 'init' else cls.__init__
    if usage == 'from':
        return undefined_parameter_action.value.handle_from_dict(cls=cls, kvs=kvs)
    elif usage == 'to':
        return undefined_parameter_action.value.handle_to_dict(obj=cls, kvs=kvs)
    elif usage == 'dump':
        return undefined_parameter_action.value.handle_dump(obj=cls)
    elif usage == 'init':
        return undefined_parameter_action.value.create_init(obj=cls)
    else:
        raise ValueError(f"usage must be one of ['to', 'from', 'dump', 'init'], but is '{usage}'")