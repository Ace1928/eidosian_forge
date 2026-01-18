from __future__ import annotations
import errno
import logging
import os
import os.path
import sys
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from io import StringIO
from textwrap import dedent
from typing import TYPE_CHECKING
from watchdog.observers.api import BaseObserverSubclassCallable
from watchdog.utils import WatchdogShutdown, load_class
from watchdog.version import VERSION_STRING
def _get_log_level_from_args(args):
    verbosity = sum(args.verbosity or [])
    if verbosity < -1:
        raise LogLevelException('-q/--quiet may be specified only once.')
    if verbosity > 2:
        raise LogLevelException('-v/--verbose may be specified up to 2 times.')
    return ['ERROR', 'WARNING', 'INFO', 'DEBUG'][1 + verbosity]