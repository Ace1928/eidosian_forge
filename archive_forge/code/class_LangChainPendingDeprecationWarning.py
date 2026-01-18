import contextlib
import functools
import inspect
import warnings
from typing import Any, Callable, Generator, Type, TypeVar, Union, cast
from langchain_core._api.internal import is_caller_internal
class LangChainPendingDeprecationWarning(PendingDeprecationWarning):
    """A class for issuing deprecation warnings for LangChain users."""