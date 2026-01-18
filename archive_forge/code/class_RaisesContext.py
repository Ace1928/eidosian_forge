import sys
import re
import functools
import os
import contextlib
import warnings
import inspect
import pathlib
from typing import Any, Callable
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.exceptions import ignore_warnings # noqa:F401
class RaisesContext:

    def __init__(self, expectedException):
        self.expectedException = expectedException

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            raise Failed('DID NOT RAISE')
        return issubclass(exc_type, self.expectedException)