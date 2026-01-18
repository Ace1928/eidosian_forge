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
class AsyncReadMode(EnvironmentVariable, type=bool):
    """
    It does not wait for the end of reading information from the source.

    It basically means, that the reading function only launches tasks for the dataframe
    to be read/created, but not ensures that the construction is finalized by the time
    the reading function returns a dataframe.

    This option was brought to improve performance of reading/construction
    of Modin DataFrames, however it may also:

    1. Increase the peak memory consumption. Since the garbage collection of the
    temporary objects created during the reading is now also lazy and will only
    be performed when the reading/construction is actually finished.

    2. Can break situations when the source is manually deleted after the reading
    function returns a result, for example, when reading inside of a context-block
    that deletes the file on ``__exit__()``.
    """
    varname = 'MODIN_ASYNC_READ_MODE'
    default = False