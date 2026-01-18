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
class IsDebug(EnvironmentVariable, type=bool):
    """Force Modin engine to be "Python" unless specified by $MODIN_ENGINE."""
    varname = 'MODIN_DEBUG'