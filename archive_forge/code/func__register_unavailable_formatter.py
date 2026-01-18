from typing import Dict, List, Optional, Type
from .. import config
from ..utils import logging
from .formatting import (
from .np_formatter import NumpyFormatter
def _register_unavailable_formatter(unavailable_error: Exception, format_type: Optional[str], aliases: Optional[List[str]]=None):
    """
    Register an unavailable Formatter object using a name and optional aliases.
    This function must be used on an Exception object that is raised when trying to get the unavailable formatter.
    """
    aliases = aliases if aliases is not None else []
    for alias in set(aliases + [format_type]):
        _FORMAT_TYPES_ALIASES_UNAVAILABLE[alias] = unavailable_error