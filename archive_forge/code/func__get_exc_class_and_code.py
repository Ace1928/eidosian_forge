from __future__ import annotations
import importlib.util
import os
import pathlib
import sys
import typing as t
from collections import defaultdict
from functools import update_wrapper
from jinja2 import BaseLoader
from jinja2 import FileSystemLoader
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException
from werkzeug.utils import cached_property
from .. import typing as ft
from ..helpers import get_root_path
from ..templating import _default_template_ctx_processor
@staticmethod
def _get_exc_class_and_code(exc_class_or_code: type[Exception] | int) -> tuple[type[Exception], int | None]:
    """Get the exception class being handled. For HTTP status codes
        or ``HTTPException`` subclasses, return both the exception and
        status code.

        :param exc_class_or_code: Any exception class, or an HTTP status
            code as an integer.
        """
    exc_class: type[Exception]
    if isinstance(exc_class_or_code, int):
        try:
            exc_class = default_exceptions[exc_class_or_code]
        except KeyError:
            raise ValueError(f"'{exc_class_or_code}' is not a recognized HTTP error code. Use a subclass of HTTPException with that code instead.") from None
    else:
        exc_class = exc_class_or_code
    if isinstance(exc_class, Exception):
        raise TypeError(f'{exc_class!r} is an instance, not a class. Handlers can only be registered for Exception classes or HTTP error codes.')
    if not issubclass(exc_class, Exception):
        raise ValueError(f"'{exc_class.__name__}' is not a subclass of Exception. Handlers can only be registered for Exception classes or HTTP error codes.")
    if issubclass(exc_class, HTTPException):
        return (exc_class, exc_class.code)
    else:
        return (exc_class, None)