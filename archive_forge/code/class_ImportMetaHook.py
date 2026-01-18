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
class ImportMetaHook:

    def __init__(self) -> None:
        self.modules: Dict[str, ModuleType] = dict()
        self.on_import: Dict[str, list] = dict()

    def add(self, fullname: str, on_import: Callable) -> None:
        self.on_import.setdefault(fullname, []).append(on_import)

    def install(self) -> None:
        sys.meta_path.insert(0, self)

    def uninstall(self) -> None:
        sys.meta_path.remove(self)

    def find_module(self, fullname: str, path: Optional[str]=None) -> Optional['ImportMetaHook']:
        if fullname in self.on_import:
            return self
        return None

    def load_module(self, fullname: str) -> ModuleType:
        self.uninstall()
        mod = importlib.import_module(fullname)
        self.install()
        self.modules[fullname] = mod
        on_imports = self.on_import.get(fullname)
        if on_imports:
            for f in on_imports:
                f()
        return mod

    def get_modules(self) -> Tuple[str, ...]:
        return tuple(self.modules)

    def get_module(self, module: str) -> ModuleType:
        return self.modules[module]