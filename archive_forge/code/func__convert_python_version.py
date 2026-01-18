import importlib.util
import logging
import os
import textwrap
from functools import partial
from optparse import SUPPRESS_HELP, Option, OptionGroup, OptionParser, Values
from textwrap import dedent
from typing import Any, Callable, Dict, Optional, Tuple
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.cli.parser import ConfigOptionParser
from pip._internal.exceptions import CommandError
from pip._internal.locations import USER_CACHE_DIR, get_src_prefix
from pip._internal.models.format_control import FormatControl
from pip._internal.models.index import PyPI
from pip._internal.models.target_python import TargetPython
from pip._internal.utils.hashes import STRONG_HASHES
from pip._internal.utils.misc import strtobool
def _convert_python_version(value: str) -> Tuple[Tuple[int, ...], Optional[str]]:
    """
    Convert a version string like "3", "37", or "3.7.3" into a tuple of ints.

    :return: A 2-tuple (version_info, error_msg), where `error_msg` is
        non-None if and only if there was a parsing error.
    """
    if not value:
        return (None, None)
    parts = value.split('.')
    if len(parts) > 3:
        return ((), 'at most three version parts are allowed')
    if len(parts) == 1:
        value = parts[0]
        if len(value) > 1:
            parts = [value[0], value[1:]]
    try:
        version_info = tuple((int(part) for part in parts))
    except ValueError:
        return ((), 'each version part must be an integer')
    return (version_info, None)