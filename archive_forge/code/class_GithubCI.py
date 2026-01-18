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
class GithubCI(EnvironmentVariable, type=bool):
    """Set to true when running Modin in GitHub CI."""
    varname = 'MODIN_GITHUB_CI'
    default = False