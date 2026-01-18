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
class CIAWSAccessKeyID(EnvironmentVariable, type=str):
    """Set to AWS_ACCESS_KEY_ID when running mock S3 tests for Modin in GitHub CI."""
    varname = 'AWS_ACCESS_KEY_ID'
    default = 'foobar_key'