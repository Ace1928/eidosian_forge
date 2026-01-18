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
class CpuCount(EnvironmentVariable, type=int):
    """How many CPU cores to use during initialization of the Modin engine."""
    varname = 'MODIN_CPUS'

    @classmethod
    def _get_default(cls) -> int:
        """
        Get default value of the config.

        Returns
        -------
        int
        """
        import multiprocessing
        return multiprocessing.cpu_count()