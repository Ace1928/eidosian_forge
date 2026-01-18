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
class LogMode(EnvironmentVariable, type=ExactStr):
    """Set ``LogMode`` value if users want to opt-in."""
    varname = 'MODIN_LOG_MODE'
    choices = ('enable', 'disable', 'enable_api_only')
    default = 'disable'

    @classmethod
    def enable(cls) -> None:
        """Enable all logging levels."""
        cls.put('enable')

    @classmethod
    def disable(cls) -> None:
        """Disable logging feature."""
        cls.put('disable')

    @classmethod
    def enable_api_only(cls) -> None:
        """Enable API level logging."""
        warnings.warn("'enable_api_only' value for LogMode is deprecated and" + 'will be removed in a future version.')
        cls.put('enable_api_only')