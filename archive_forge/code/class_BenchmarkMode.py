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
class BenchmarkMode(EnvironmentVariable, type=bool):
    """Whether or not to perform computations synchronously."""
    varname = 'MODIN_BENCHMARK_MODE'
    default = False

    @classmethod
    def put(cls, value: bool) -> None:
        """
        Set ``BenchmarkMode`` value only if progress bar feature is disabled.

        Parameters
        ----------
        value : bool
            Config value to set.
        """
        if value and ProgressBar.get():
            raise ValueError("BenchmarkMode isn't compatible with ProgressBar")
        super().put(value)