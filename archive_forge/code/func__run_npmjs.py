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
def _run_npmjs(argv: list[str], input: dict[str, Any] | None=None) -> str:
    return _run(_npmjs_path(), argv, input)