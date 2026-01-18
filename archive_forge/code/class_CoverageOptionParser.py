from __future__ import annotations
import glob
import optparse     # pylint: disable=deprecated-module
import os
import os.path
import shlex
import sys
import textwrap
import traceback
from typing import cast, Any, NoReturn
import coverage
from coverage import Coverage
from coverage import env
from coverage.collector import HAS_CTRACER
from coverage.config import CoverageConfig
from coverage.control import DEFAULT_DATAFILE
from coverage.data import combinable_files, debug_data_file
from coverage.debug import info_header, short_stack, write_formatted_info
from coverage.exceptions import _BaseCoverageException, _ExceptionDuringRun, NoSource
from coverage.execfile import PyRunner
from coverage.results import Numbers, should_fail_under
from coverage.version import __url__
class CoverageOptionParser(optparse.OptionParser):
    """Base OptionParser for coverage.py.

    Problems don't exit the program.
    Defaults are initialized for all options.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs['add_help_option'] = False
        super().__init__(*args, **kwargs)
        self.set_defaults(action=None, append=None, branch=None, concurrency=None, context=None, contexts=None, data_file=None, debug=None, directory=None, fail_under=None, format=None, help=None, ignore_errors=None, include=None, keep=None, module=None, omit=None, parallel_mode=None, precision=None, pylib=None, quiet=None, rcfile=True, show_contexts=None, show_missing=None, skip_covered=None, skip_empty=None, sort=None, source=None, timid=None, title=None, version=None)
        self.disable_interspersed_args()

    class OptionParserError(Exception):
        """Used to stop the optparse error handler ending the process."""
        pass

    def parse_args_ok(self, args: list[str]) -> tuple[bool, optparse.Values | None, list[str]]:
        """Call optparse.parse_args, but return a triple:

        (ok, options, args)

        """
        try:
            options, args = super().parse_args(args)
        except self.OptionParserError:
            return (False, None, [])
        return (True, options, args)

    def error(self, msg: str) -> NoReturn:
        """Override optparse.error so sys.exit doesn't get called."""
        show_help(msg)
        raise self.OptionParserError