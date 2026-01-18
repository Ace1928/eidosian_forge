from __future__ import annotations
import re
import os
import sys
import logging
import typing
import traceback
import warnings
import pprint
import atexit as _atexit
import functools
import threading
from enum import Enum
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from typing import Type, Union, Optional, Any, List, Dict, Tuple, Callable, Set, TYPE_CHECKING
def _format_item(self, msg: Any, max_length: Optional[int]=None, colored: Optional[bool]=False, level: Optional[str]=None, _is_part: Optional[bool]=False) -> str:
    """
        Formats an item
        """
    if isinstance(msg, str):
        return msg[:max_length] if max_length else msg
    if isinstance(msg, (float, int, bool, type(None))):
        return str(msg)[:max_length] if max_length else str(msg)
    if isinstance(msg, (list, set)):
        _msg = str(msg) if _is_part else '\n' + ''.join((f'- {item}\n' for item in msg))
        return _msg[:max_length] if max_length else _msg
    prefix, suffix = ('', '')
    if colored:
        prefix = '|g|'
        if level:
            level = level.lower()
            prefix = DEFAULT_STATUS_COLORS.get(level, '|g|')
        suffix = '|e|'
    if isinstance(msg, dict):
        _msg = '\n'
        for key, value in msg.items():
            _value = f'{value}'
            if max_length and len(_value) > max_length:
                _value = f'{_value[:max_length]}...'
            _msg += f'- {prefix}{key}{suffix}: {_value}\n'
        return _msg.rstrip()
    if isinstance(msg, tuple):
        _msg = ''.join((f'- {prefix}{key}{suffix}: {self._format_item(value, max_length=max_length, colored=colored, level=level, _is_part=True)}\n' for key, value in zip(msg[0], msg[1])))
        return _msg[:max_length] if max_length else _msg
    if hasattr(msg, 'dict') and hasattr(msg, 'Config') or hasattr(msg, 'fields'):
        _msg = f'{prefix}[{msg.__class__.__name__}]{suffix}'
        fields = msg.fields.keys() if hasattr(msg, 'fields') else msg.__fields__.keys()
        for field in fields:
            field_str = f'{prefix}{field}{suffix}'
            val_s = f'\n  {field_str}: {getattr(msg, field)!r}'
            if max_length is not None and len(val_s) > max_length:
                val_s = f'{val_s[:max_length]}...'
            _msg += val_s
        return _msg
    if hasattr(msg, 'model_dump'):
        return self._format_item(msg.model_dump(mode='json'), max_length=max_length, colored=colored, level=level, _is_part=_is_part)
    if hasattr(msg, 'dict'):
        return self._format_item(msg.dict(), max_length=max_length, colored=colored, level=level, _is_part=_is_part)
    if hasattr(msg, 'json'):
        return self._format_item(msg.json(), max_length=max_length, colored=colored, level=level, _is_part=_is_part)
    if hasattr(msg, '__dict__'):
        return self._format_item(msg.__dict__, max_length=max_length, colored=colored, level=level, _is_part=_is_part)
    return str(msg)[:max_length] if max_length else str(msg)