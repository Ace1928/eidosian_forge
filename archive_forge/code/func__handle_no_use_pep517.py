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
def _handle_no_use_pep517(option: Option, opt: str, value: str, parser: OptionParser) -> None:
    """
    Process a value provided for the --no-use-pep517 option.

    This is an optparse.Option callback for the no_use_pep517 option.
    """
    if value is not None:
        msg = 'A value was passed for --no-use-pep517,\n        probably using either the PIP_NO_USE_PEP517 environment variable\n        or the "no-use-pep517" config file option. Use an appropriate value\n        of the PIP_USE_PEP517 environment variable or the "use-pep517"\n        config file option instead.\n        '
        raise_option_error(parser, option=option, msg=msg)
    packages = ('setuptools', 'wheel')
    if not all((importlib.util.find_spec(package) for package in packages)):
        msg = f'It is not possible to use --no-use-pep517 without {' and '.join(packages)} installed.'
        raise_option_error(parser, option=option, msg=msg)
    parser.values.use_pep517 = False