import contextlib
import functools
import inspect
import warnings
from typing import Any, Callable, Generator, Type, TypeVar, Union, cast
from langchain_core._api.internal import is_caller_internal
Wrap the wrapped function using the wrapper and update the docstring.

                Args:
                    wrapper: The wrapper function.
                    new_doc: The new docstring.

                Returns:
                    The wrapped function.
                