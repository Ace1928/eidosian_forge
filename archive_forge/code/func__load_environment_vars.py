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
def _load_environment_vars(self) -> None:
    """Loads configuration from environment variables"""
    self._config[kinds.ENV_VAR].update(self._normalized_keys(':env:', self.get_environ_vars()))