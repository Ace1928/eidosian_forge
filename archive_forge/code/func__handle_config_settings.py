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
def _handle_config_settings(option: Option, opt_str: str, value: str, parser: OptionParser) -> None:
    key, sep, val = value.partition('=')
    if sep != '=':
        parser.error(f'Arguments to {opt_str} must be of the form KEY=VAL')
    dest = getattr(parser.values, option.dest)
    if dest is None:
        dest = {}
        setattr(parser.values, option.dest, dest)
    if key in dest:
        if isinstance(dest[key], list):
            dest[key].append(val)
        else:
            dest[key] = [dest[key], val]
    else:
        dest[key] = val