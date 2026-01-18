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
def _device_agnostic_dispatch(device: str, dispatch_table: Dict[str, Callable], *args, **kwargs):
    if device not in dispatch_table:
        return dispatch_table['default'](*args, **kwargs)
    fn = dispatch_table[device]
    if fn is None:
        return None
    return fn(*args, **kwargs)