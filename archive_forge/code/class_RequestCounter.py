import collections
import contextlib
import doctest
import functools
import importlib
import inspect
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union
from unittest import mock
from unittest.mock import patch
import urllib3
from transformers import logging as transformers_logging
from .integrations import (
from .integrations.deepspeed import is_deepspeed_available
from .utils import (
import asyncio  # noqa
class RequestCounter:
    """
    Helper class that will count all requests made online.

    Might not be robust if urllib3 changes its logging format but should be good enough for us.

    Usage:
    ```py
    with RequestCounter() as counter:
        _ = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    assert counter["GET"] == 0
    assert counter["HEAD"] == 1
    assert counter.total_calls == 1
    ```
    """

    def __enter__(self):
        self._counter = defaultdict(int)
        self.patcher = patch.object(urllib3.connectionpool.log, 'debug', wraps=urllib3.connectionpool.log.debug)
        self.mock = self.patcher.start()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        for call in self.mock.call_args_list:
            log = call.args[0] % call.args[1:]
            for method in ('HEAD', 'GET', 'POST', 'PUT', 'DELETE', 'CONNECT', 'OPTIONS', 'TRACE', 'PATCH'):
                if method in log:
                    self._counter[method] += 1
                    break
        self.patcher.stop()

    def __getitem__(self, key: str) -> int:
        return self._counter[key]

    @property
    def total_calls(self) -> int:
        return sum(self._counter.values())