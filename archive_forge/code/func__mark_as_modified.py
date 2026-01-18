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
def _mark_as_modified(self, fname: str, parser: RawConfigParser) -> None:
    file_parser_tuple = (fname, parser)
    if file_parser_tuple not in self._modified_parsers:
        self._modified_parsers.append(file_parser_tuple)