from __future__ import annotations
import collections.abc
import sys
import textwrap
import traceback
from functools import singledispatch
from types import TracebackType
from typing import Any, List, Optional
from ._exceptions import BaseExceptionGroup
Format the exception part of the traceback.
        The return value is a generator of strings, each ending in a newline.
        Normally, the generator emits a single string; however, for
        SyntaxError exceptions, it emits several lines that (when
        printed) display detailed information about where the syntax
        error occurred.
        The message indicating which exception occurred is always the last
        string in the output.
        