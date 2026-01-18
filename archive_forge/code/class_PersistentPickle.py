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
class PersistentPickle(EnvironmentVariable, type=bool):
    """Whether serialization should be persistent."""
    varname = 'MODIN_PERSISTENT_PICKLE'
    default = False