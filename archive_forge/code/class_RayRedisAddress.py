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
class RayRedisAddress(EnvironmentVariable, type=ExactStr):
    """Redis address to connect to when running in Ray cluster."""
    varname = 'MODIN_REDIS_ADDRESS'