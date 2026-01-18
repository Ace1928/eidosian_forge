import contextlib
import functools
import inspect
import warnings
from typing import Any, Callable, Generator, Type, TypeVar, Union, cast
from langchain_core._api.internal import is_caller_internal
class LangChainBetaWarning(DeprecationWarning):
    """A class for issuing beta warnings for LangChain users."""