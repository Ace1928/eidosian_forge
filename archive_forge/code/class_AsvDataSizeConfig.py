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
class AsvDataSizeConfig(EnvironmentVariable, type=ExactStr):
    """Allows to override default size of data (shapes)."""
    varname = 'MODIN_ASV_DATASIZE_CONFIG'
    default = None