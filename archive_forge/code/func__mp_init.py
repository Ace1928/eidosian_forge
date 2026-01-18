from __future__ import annotations
import argparse
import contextlib
import errno
import logging
import multiprocessing.pool
import operator
import signal
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from flake8 import defaults
from flake8 import exceptions
from flake8 import processor
from flake8 import utils
from flake8._compat import FSTRING_START
from flake8.discover_files import expand_paths
from flake8.options.parse_args import parse_args
from flake8.plugins.finder import Checkers
from flake8.plugins.finder import LoadedPlugin
from flake8.style_guide import StyleGuideManager
def _mp_init(argv: Sequence[str]) -> None:
    global _mp_plugins, _mp_options
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        (_mp_plugins, _mp_options)
    except NameError:
        plugins, options = parse_args(argv)
        _mp_plugins, _mp_options = (plugins.checkers, options)