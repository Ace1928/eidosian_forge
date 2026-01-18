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
class LazyExecution(EnvironmentVariable, type=str):
    """
    Lazy execution mode.

    Supported values:
        `Auto` - the execution mode is chosen by the engine for each operation (default value).
        `On`   - the lazy execution is performed wherever it's possible.
        `Off`  - the lazy execution is disabled.
    """
    varname = 'MODIN_LAZY_EXECUTION'
    choices = ('Auto', 'On', 'Off')
    default = 'Auto'