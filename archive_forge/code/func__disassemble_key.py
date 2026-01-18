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
def _disassemble_key(name: str) -> List[str]:
    if '.' not in name:
        error_message = f"Key does not contain dot separated section and key. Perhaps you wanted to use 'global.{name}' instead?"
        raise ConfigurationError(error_message)
    return name.split('.', 1)