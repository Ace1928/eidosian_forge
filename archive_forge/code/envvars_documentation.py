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

        Assign a value to the DocModule config.

        Parameters
        ----------
        value : str
            Config value to set.
        