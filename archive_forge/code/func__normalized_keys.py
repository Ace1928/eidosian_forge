import configparser
import locale
import os
import sys
from typing import Any, Dict, Iterable, List, NewType, Optional, Tuple
from pip._internal.exceptions import (
from pip._internal.utils import appdirs
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import ensure_dir, enum
def _normalized_keys(self, section: str, items: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    """Normalizes items to construct a dictionary with normalized keys.

        This routine is where the names become keys and are made the same
        regardless of source - configuration files or environment.
        """
    normalized = {}
    for name, val in items:
        key = section + '.' + _normalize_name(name)
        normalized[key] = val
    return normalized