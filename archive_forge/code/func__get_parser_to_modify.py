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
def _get_parser_to_modify(self) -> Tuple[str, RawConfigParser]:
    assert self.load_only
    parsers = self._parsers[self.load_only]
    if not parsers:
        raise ConfigurationError('Fatal Internal error [id=2]. Please report as a bug.')
    return parsers[-1]