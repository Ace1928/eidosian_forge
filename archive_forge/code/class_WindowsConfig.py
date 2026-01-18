from __future__ import annotations
import abc
import dataclasses
import enum
import os
import pickle
import sys
import typing as t
from .constants import (
from .io import (
from .completion import (
from .util import (
@dataclasses.dataclass
class WindowsConfig(HostConfig, metaclass=abc.ABCMeta):
    """Base class for Windows host configuration."""