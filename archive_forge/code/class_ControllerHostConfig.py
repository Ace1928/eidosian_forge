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
class ControllerHostConfig(PosixConfig, metaclass=abc.ABCMeta):
    """Base class for host configurations which support the controller."""

    @abc.abstractmethod
    def get_default_targets(self, context: HostContext) -> list[ControllerConfig]:
        """Return the default targets for this host config."""