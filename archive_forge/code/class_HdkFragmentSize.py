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
class HdkFragmentSize(EnvironmentVariable, type=int):
    """How big a fragment in HDK should be when creating a table (in rows)."""
    varname = 'MODIN_HDK_FRAGMENT_SIZE'