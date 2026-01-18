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
class NPartitions(EnvironmentVariable, type=int):
    """How many partitions to use for a Modin DataFrame (along each axis)."""
    varname = 'MODIN_NPARTITIONS'

    @classmethod
    def _put(cls, value: int) -> None:
        """
        Put specific value if NPartitions wasn't set by a user yet.

        Parameters
        ----------
        value : int
            Config value to set.

        Notes
        -----
        This method is used to set NPartitions from cluster resources internally
        and should not be called by a user.
        """
        if cls.get_value_source() == ValueSource.DEFAULT:
            cls.put(value)

    @classmethod
    def _get_default(cls) -> int:
        """
        Get default value of the config.

        Returns
        -------
        int
        """
        if StorageFormat.get() == 'Cudf':
            return GpuCount.get()
        else:
            return CpuCount.get()