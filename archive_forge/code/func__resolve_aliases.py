import colorsys
import contextlib
import dataclasses
import functools
import gzip
import importlib
import importlib.util
import itertools
import json
import logging
import math
import numbers
import os
import platform
import queue
import random
import re
import secrets
import shlex
import socket
import string
import sys
import tarfile
import tempfile
import threading
import time
import types
import urllib
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from importlib import import_module
from sys import getsizeof
from types import ModuleType
from typing import (
import requests
import yaml
import wandb
import wandb.env
from wandb.errors import AuthenticationError, CommError, UsageError, term
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, runid
from wandb.sdk.lib.json_util import dump, dumps
from wandb.sdk.lib.paths import FilePathStr, StrPath
def _resolve_aliases(aliases: Optional[Union[str, Iterable[str]]]) -> List[str]:
    """Add the 'latest' alias and ensure that all aliases are unique.

    Takes in `aliases` which can be None, str, or List[str] and returns List[str].
    Ensures that "latest" is always present in the returned list.

    Args:
        aliases: `Optional[Union[str, List[str]]]`

    Returns:
        List[str], with "latest" always present.

    Usage:

    ```python
    aliases = _resolve_aliases(["best", "dev"])
    assert aliases == ["best", "dev", "latest"]

    aliases = _resolve_aliases("boom")
    assert aliases == ["boom", "latest"]
    ```
    """
    aliases = aliases or ['latest']
    if isinstance(aliases, str):
        aliases = [aliases]
    try:
        return list(set(aliases) | {'latest'})
    except TypeError as exc:
        raise ValueError('`aliases` must be Iterable or None') from exc