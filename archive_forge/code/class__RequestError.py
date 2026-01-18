from __future__ import annotations
import io
import json
from email.parser import Parser
from importlib.resources import files
from typing import TYPE_CHECKING, Any
import js  # type: ignore[import-not-found]
from pyodide.ffi import (  # type: ignore[import-not-found]
from .request import EmscriptenRequest
from .response import EmscriptenResponse
class _RequestError(Exception):

    def __init__(self, message: str | None=None, *, request: EmscriptenRequest | None=None, response: EmscriptenResponse | None=None):
        self.request = request
        self.response = response
        self.message = message
        super().__init__(self.message)