from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional,
from .pyutils import get_named_object
def _add_help_and_info(self, key: K, help=None, info=None):
    """Add the help and information about this key"""
    self._help_dict[key] = help
    self._info_dict[key] = info