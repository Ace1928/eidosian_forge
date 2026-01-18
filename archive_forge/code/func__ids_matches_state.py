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
def _ids_matches_state(self, expected_states, ids, timeout=0.5):
    states = self.query_task_states(ids, timeout=timeout)
    return all((any((t in s for s in [states[k] for k in expected_states])) for t in ids))