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
def _prompt_choice(input_timeout: Union[int, float, None]=None, jupyter: bool=False) -> str:
    input_fn: Callable = input
    prompt = term.LOG_STRING
    if input_timeout is not None:
        from wandb.sdk.lib import timed_input
        input_fn = functools.partial(timed_input.timed_input, timeout=input_timeout)
        if platform.system() == 'Windows':
            prompt = 'wandb'
    text = f'{prompt}: Enter your choice: '
    if input_fn == input:
        choice = input_fn(text)
    else:
        choice = input_fn(text, jupyter=jupyter)
    return choice