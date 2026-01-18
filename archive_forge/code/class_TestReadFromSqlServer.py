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
class TestReadFromSqlServer(EnvironmentVariable, type=bool):
    """Set to true to test reading from SQL server."""
    varname = 'MODIN_TEST_READ_FROM_SQL_SERVER'
    default = False