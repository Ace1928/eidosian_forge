import os
import re
import sys
from collections import namedtuple
from . import local
def _find_option_with_arg(argv, short_opts=None, long_opts=None):
    """Search argv for options specifying short and longopt alternatives.

    Returns:
        str: value for option found
    Raises:
        KeyError: if option not found.
    """
    for i, arg in enumerate(argv):
        if arg.startswith('-'):
            if long_opts and arg.startswith('--'):
                name, sep, val = arg.partition('=')
                if name in long_opts:
                    return val if sep else argv[i + 1]
            if short_opts and arg in short_opts:
                return argv[i + 1]
    raise KeyError('|'.join(short_opts or [] + long_opts or []))