from __future__ import annotations
import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread
from typing import (
from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable
class PytestTracker:
    """Track the current pytest test name to add to debug messages."""

    def __init__(self) -> None:
        self.test_name: str | None = None

    def filter(self, text: str) -> str:
        """Add a message when the pytest test changes."""
        test_name = os.getenv('PYTEST_CURRENT_TEST')
        if test_name != self.test_name:
            text = f'Pytest context: {test_name}\n' + text
            self.test_name = test_name
        return text