import socket
import sys
from collections import defaultdict
from functools import partial
from itertools import count
from typing import Any, Callable, Dict, Sequence, TextIO, Tuple  # noqa
from kombu.exceptions import ContentDisallowed
from kombu.utils.functional import retry_over_time
from celery import states
from celery.exceptions import TimeoutError
from celery.result import AsyncResult, ResultSet  # noqa
from celery.utils.text import truncate
from celery.utils.time import humanize_seconds as _humanize_seconds
def _init_manager(self, block_timeout=30 * 60.0, no_join=False, stdout=None, stderr=None):
    self.stdout = sys.stdout if stdout is None else stdout
    self.stderr = sys.stderr if stderr is None else stderr
    self.connerrors = self.app.connection().recoverable_connection_errors
    self.block_timeout = block_timeout
    self.no_join = no_join