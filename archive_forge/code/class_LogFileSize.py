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
class LogFileSize(EnvironmentVariable, type=int):
    """Max size of logs (in MBs) to store per Modin job."""
    varname = 'MODIN_LOG_FILE_SIZE'
    default = 10

    @classmethod
    def put(cls, value: int) -> None:
        """
        Set ``LogFileSize`` with extra checks.

        Parameters
        ----------
        value : int
            Config value to set.
        """
        if value <= 0:
            raise ValueError(f'Log file size should be > 0 MB, passed value {value}')
        super().put(value)

    @classmethod
    def get(cls) -> int:
        """
        Get ``LogFileSize`` with extra checks.

        Returns
        -------
        int
        """
        log_file_size = super().get()
        assert log_file_size > 0, '`LogFileSize` should be > 0'
        return log_file_size