import importlib
import os
import secrets
import sys
import warnings
from textwrap import dedent
from typing import Any, Optional
from packaging import version
from pandas.util._decorators import doc  # type: ignore[attr-defined]
from modin.config.pubsub import (
class LogMemoryInterval(EnvironmentVariable, type=int):
    """Interval (in seconds) to profile memory utilization for logging."""
    varname = 'MODIN_LOG_MEMORY_INTERVAL'
    default = 5

    @classmethod
    def put(cls, value: int) -> None:
        """
        Set ``LogMemoryInterval`` with extra checks.

        Parameters
        ----------
        value : int
            Config value to set.
        """
        if value <= 0:
            raise ValueError(f'Log memory Interval should be > 0, passed value {value}')
        super().put(value)

    @classmethod
    def get(cls) -> int:
        """
        Get ``LogMemoryInterval`` with extra checks.

        Returns
        -------
        int
        """
        log_memory_interval = super().get()
        assert log_memory_interval > 0, '`LogMemoryInterval` should be > 0'
        return log_memory_interval