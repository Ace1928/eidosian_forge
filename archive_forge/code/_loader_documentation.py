from __future__ import annotations
import importlib.metadata as importlib_metadata
import warnings
from typing import TYPE_CHECKING, Final, Iterable
Load plugins for Pydantic.

    Inspired by: https://github.com/pytest-dev/pluggy/blob/1.3.0/src/pluggy/_manager.py#L376-L402
    