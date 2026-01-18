from __future__ import annotations
import logging # isort:skip
import hashlib
import json
import os
import re
import sys
from os.path import (
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Callable, Sequence
from ..core.has_props import HasProps
from ..settings import settings
from .strings import snakify
class Implementation:
    """ Base class for representing Bokeh custom model implementations.

    """
    file: str | None = None
    code: str

    @property
    def lang(self) -> str:
        raise NotImplementedError()