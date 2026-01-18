from __future__ import annotations
import argparse
import contextlib
import copy
import enum
import functools
import logging
from typing import Generator
from typing import Sequence
from flake8 import defaults
from flake8 import statistics
from flake8 import utils
from flake8.formatting import base as base_formatter
from flake8.violation import Violation
def _explicitly_chosen(*, option: list[str] | None, extend: list[str] | None) -> tuple[str, ...]:
    ret = [*(option or []), *(extend or [])]
    return tuple(sorted(ret, reverse=True))