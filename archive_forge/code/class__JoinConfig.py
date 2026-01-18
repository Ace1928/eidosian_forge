import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, NamedTuple, Optional, Type
import torch
import torch.distributed as dist
class _JoinConfig(NamedTuple):
    """This includes all fields needed from a :class:`Joinable` instance for the join context manager side."""
    enable: bool
    throw_on_early_termination: bool
    is_first_joinable: bool

    @staticmethod
    def construct_disabled_join_config():
        """Return a :class:`_JoinConfig` instance indicating that join-related logic should be disabled.

        e.g. if the caller is not in a join context manager.
        """
        return _JoinConfig(enable=False, throw_on_early_termination=False, is_first_joinable=False)