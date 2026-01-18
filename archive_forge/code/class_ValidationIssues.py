from __future__ import annotations
import logging # isort:skip
import contextlib
from typing import (
from ...model import Model
from ...settings import settings
from ...util.dataclasses import dataclass, field
from .issue import Warning
@dataclass
class ValidationIssues:
    error: list[ValidationIssue] = field(default_factory=list)
    warning: list[ValidationIssue] = field(default_factory=list)