from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class ParserMode(enum.Enum):
    """Mode the parser is operating in."""
    PARSE = enum.auto()
    COMPLETE = enum.auto()
    LIST = enum.auto()