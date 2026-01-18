from typing import Dict, List, Optional, Type
from .. import config
from ..utils import logging
from .formatting import (
from .np_formatter import NumpyFormatter
def _register_formatter(formatter_cls: type, format_type: Optional[str], aliases: Optional[List[str]]=None):
    """
    Register a Formatter object using a name and optional aliases.
    This function must be used on a Formatter class.
    """
    aliases = aliases if aliases is not None else []
    if format_type in _FORMAT_TYPES:
        logger.warning(f"Overwriting format type '{format_type}' ({_FORMAT_TYPES[format_type].__name__} -> {formatter_cls.__name__})")
    _FORMAT_TYPES[format_type] = formatter_cls
    for alias in set(aliases + [format_type]):
        if alias in _FORMAT_TYPES_ALIASES:
            logger.warning(f"Overwriting format type alias '{alias}' ({_FORMAT_TYPES_ALIASES[alias]} -> {format_type})")
        _FORMAT_TYPES_ALIASES[alias] = format_type