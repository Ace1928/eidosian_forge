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
class ExperimentalNumPyAPI(EnvWithSibilings, type=bool):
    """
    Set to true to use Modin's implementation of NumPy API.

    This parameter is deprecated. Use ``ModinNumpy`` instead.
    """
    varname = 'MODIN_EXPERIMENTAL_NUMPY_API'
    default = False

    @classmethod
    def _sibling(cls) -> type[EnvWithSibilings]:
        """Get a parameter sibling."""
        return ModinNumpy