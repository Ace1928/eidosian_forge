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
def _ensure_have_load_only(self) -> None:
    if self.load_only is None:
        raise ConfigurationError('Needed a specific file to be modifying.')
    logger.debug('Will be working with %s variant only', self.load_only)