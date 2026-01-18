from __future__ import annotations
import typing as t
from jinja2.loaders import BaseLoader
from werkzeug.routing import RequestRedirect
from .blueprints import Blueprint
from .globals import request_ctx
from .sansio.app import App
class UnexpectedUnicodeError(AssertionError, UnicodeError):
    """Raised in places where we want some better error reporting for
    unexpected unicode or binary data.
    """