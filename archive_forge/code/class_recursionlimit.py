from __future__ import annotations
import os
import sys
import unittest
import textwrap
from typing import Any
from . import markdown, Markdown, util
class recursionlimit:
    """
    A context manager which temporarily modifies the Python recursion limit.

    The testing framework, coverage, etc. may add an arbitrary number of levels to the depth. To maintain consistency
    in the tests, the current stack depth is determined when called, then added to the provided limit.

    Example usage:

    ``` python
    with recursionlimit(20):
        # test code here
    ```

    See <https://stackoverflow.com/a/50120316/866026>.
    """

    def __init__(self, limit):
        self.limit = util._get_stack_depth() + limit
        self.old_limit = sys.getrecursionlimit()

    def __enter__(self):
        sys.setrecursionlimit(self.limit)

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)