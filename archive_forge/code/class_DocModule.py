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
class DocModule(EnvironmentVariable, type=ExactStr):
    """
    The module to use that will be used for docstrings.

    The value set here must be a valid, importable module. It should have
    a `DataFrame`, `Series`, and/or several APIs directly (e.g. `read_csv`).
    """
    varname = 'MODIN_DOC_MODULE'
    default = 'pandas'

    @classmethod
    def put(cls, value: str) -> None:
        """
        Assign a value to the DocModule config.

        Parameters
        ----------
        value : str
            Config value to set.
        """
        super().put(value)
        import modin.pandas as pd
        importlib.reload(pd.accessor)
        importlib.reload(pd.base)
        importlib.reload(pd.dataframe)
        importlib.reload(pd.general)
        importlib.reload(pd.groupby)
        importlib.reload(pd.io)
        importlib.reload(pd.iterator)
        importlib.reload(pd.series)
        importlib.reload(pd.series_utils)
        importlib.reload(pd.utils)
        importlib.reload(pd.window)
        importlib.reload(pd)