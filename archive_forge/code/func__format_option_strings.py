import logging
import optparse
import shutil
import sys
import textwrap
from contextlib import suppress
from typing import Any, Dict, Generator, List, Tuple
from pip._internal.cli.status_codes import UNKNOWN_ERROR
from pip._internal.configuration import Configuration, ConfigurationError
from pip._internal.utils.misc import redact_auth_from_url, strtobool
def _format_option_strings(self, option: optparse.Option, mvarfmt: str=' <{}>', optsep: str=', ') -> str:
    """
        Return a comma-separated list of option strings and metavars.

        :param option:  tuple of (short opt, long opt), e.g: ('-f', '--format')
        :param mvarfmt: metavar format string
        :param optsep:  separator
        """
    opts = []
    if option._short_opts:
        opts.append(option._short_opts[0])
    if option._long_opts:
        opts.append(option._long_opts[0])
    if len(opts) > 1:
        opts.insert(1, optsep)
    if option.takes_value():
        assert option.dest is not None
        metavar = option.metavar or option.dest.lower()
        opts.append(mvarfmt.format(metavar.lower()))
    return ''.join(opts)