from __future__ import annotations
import asyncio
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import traceback
import uuid
import warnings
from contextvars import copy_context
from typing import (
from langsmith import client as ls_client
from langsmith import run_trees, utils
from langsmith._internal import _aiter as aitertools
@runtime_checkable
class SupportsLangsmithExtra(Protocol, Generic[R]):
    """Implementations of this Protoc accept an optional langsmith_extra parameter.

    Args:
        *args: Variable length arguments.
        langsmith_extra (Optional[LangSmithExtra): Optional dictionary of
            additional parameters for Langsmith.
        **kwargs: Keyword arguments.

    Returns:
        R: The return type of the callable.
    """

    def __call__(self, *args: Any, langsmith_extra: Optional[LangSmithExtra]=None, **kwargs: Any) -> R:
        """Call the instance when it is called as a function.

        Args:
            *args: Variable length argument list.
            langsmith_extra: Optional dictionary containing additional
                parameters specific to Langsmith.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            R: The return value of the method.

        """
        ...