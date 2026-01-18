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
class DoUseCalcite(EnvironmentVariable, type=bool):
    """Whether to use Calcite for HDK queries execution."""
    varname = 'MODIN_USE_CALCITE'
    default = True