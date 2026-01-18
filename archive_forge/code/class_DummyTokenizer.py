import functools
import importlib
import importlib.util
import inspect
import itertools
import logging
import os
import pkgutil
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import (
import catalogue
import langcodes
import numpy
import srsly
import thinc
from catalogue import Registry, RegistryError
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from thinc.api import (
from thinc.api import compounding, decaying, fix_random_seed  # noqa: F401
from . import about
from .compat import CudaStream, cupy, importlib_metadata, is_windows
from .errors import OLD_MODEL_SHORTCUTS, Errors, Warnings
from .symbols import ORTH
class DummyTokenizer:

    def __call__(self, text):
        raise NotImplementedError

    def pipe(self, texts, **kwargs):
        for text in texts:
            yield self(text)

    def to_bytes(self, **kwargs):
        return b''

    def from_bytes(self, data: bytes, **kwargs) -> 'DummyTokenizer':
        return self

    def to_disk(self, path: Union[str, Path], **kwargs) -> None:
        return None

    def from_disk(self, path: Union[str, Path], **kwargs) -> 'DummyTokenizer':
        return self