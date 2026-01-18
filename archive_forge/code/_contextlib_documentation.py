import functools
import inspect
import warnings
import sys
from typing import Any, Callable, TypeVar, cast
Allow a context manager to be used as a decorator without parentheses.