import inspect
import sys
from datetime import datetime, timezone
from collections import Counter
from typing import (Collection, Mapping, Optional, TypeVar, Any, Type, Tuple,
def _undefined_parameter_action_safe(cls):
    try:
        if cls.dataclass_json_config is None:
            return
        action_enum = cls.dataclass_json_config['undefined']
    except (AttributeError, KeyError):
        return
    if action_enum is None or action_enum.value is None:
        return
    return action_enum