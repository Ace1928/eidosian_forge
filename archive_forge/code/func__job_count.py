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
def _job_count(self) -> int:
    if utils.is_using_stdin(self.options.filenames):
        LOG.warning('The --jobs option is not compatible with supplying input using - . Ignoring --jobs arguments.')
        return 0
    jobs = self.options.jobs
    if jobs.is_auto:
        try:
            return multiprocessing.cpu_count()
        except NotImplementedError:
            return 0
    return jobs.n_jobs