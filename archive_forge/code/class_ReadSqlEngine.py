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
class ReadSqlEngine(EnvironmentVariable, type=str):
    """Engine to run `read_sql`."""
    varname = 'MODIN_READ_SQL_ENGINE'
    default = 'Pandas'
    choices = ('Pandas', 'Connectorx')