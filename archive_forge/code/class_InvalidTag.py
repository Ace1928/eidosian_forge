from __future__ import annotations
import typing
from cryptography.hazmat.bindings._rust import exceptions as rust_exceptions
class InvalidTag(Exception):
    pass