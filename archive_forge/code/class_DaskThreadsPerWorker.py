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
class DaskThreadsPerWorker(EnvironmentVariable, type=int):
    """Number of threads per Dask worker."""
    varname = 'MODIN_DASK_THREADS_PER_WORKER'
    default = 1