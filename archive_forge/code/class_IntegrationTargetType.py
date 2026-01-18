from __future__ import annotations
import collections
import collections.abc as c
import enum
import os
import re
import itertools
import abc
import typing as t
from .encoding import (
from .io import (
from .util import (
from .data import (
class IntegrationTargetType(enum.Enum):
    """Type of integration test target."""
    CONTROLLER = enum.auto()
    TARGET = enum.auto()
    UNKNOWN = enum.auto()
    CONFLICT = enum.auto()