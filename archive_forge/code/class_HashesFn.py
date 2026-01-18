from __future__ import annotations
import logging  # isort:skip
import json
import os
import re
from os.path import relpath
from pathlib import Path
from typing import (
from . import __version__
from .core.templates import CSS_RESOURCES, JS_RESOURCES
from .core.types import ID, PathLike
from .model import Model
from .settings import LogLevel, settings
from .util.dataclasses import dataclass, field
from .util.paths import ROOT_DIR
from .util.token import generate_session_id
from .util.version import is_full_release
class HashesFn(Protocol):

    @staticmethod
    def __call__(components: list[str], kind: Kind) -> Hashes:
        ...