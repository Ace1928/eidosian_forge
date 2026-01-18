from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
@dataclasses.dataclass
class DocumentationState:
    """State of the composite argument parser's generated documentation."""
    sections: dict[str, str] = dataclasses.field(default_factory=dict)