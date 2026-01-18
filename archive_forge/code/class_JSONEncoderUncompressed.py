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
class JSONEncoderUncompressed(json.JSONEncoder):
    """A JSON Encoder that handles some extra types.

    This encoder turns numpy like objects with a size > 32 into histograms.
    """

    def default(self, obj: Any) -> Any:
        if is_numpy_array(obj):
            return obj.tolist()
        elif np and isinstance(obj, np.generic):
            obj = obj.item()
        return json.JSONEncoder.default(self, obj)