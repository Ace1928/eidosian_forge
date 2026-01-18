import ast
import dataclasses
import inspect
from inspect import CO_VARARGS
from inspect import CO_VARKEYWORDS
from io import StringIO
import os
from pathlib import Path
import re
import sys
import traceback
from traceback import format_exception_only
from types import CodeType
from types import FrameType
from types import TracebackType
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Final
from typing import final
from typing import Generic
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import SupportsIndex
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import pluggy
import _pytest
from _pytest._code.source import findsource
from _pytest._code.source import getrawcode
from _pytest._code.source import getstatementrange_ast
from _pytest._code.source import Source
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import safeformat
from _pytest._io.saferepr import saferepr
from _pytest.compat import get_real_func
from _pytest.deprecated import check_ispytest
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
def _truncate_recursive_traceback(self, traceback: Traceback) -> Tuple[Traceback, Optional[str]]:
    """Truncate the given recursive traceback trying to find the starting
        point of the recursion.

        The detection is done by going through each traceback entry and
        finding the point in which the locals of the frame are equal to the
        locals of a previous frame (see ``recursionindex()``).

        Handle the situation where the recursion process might raise an
        exception (for example comparing numpy arrays using equality raises a
        TypeError), in which case we do our best to warn the user of the
        error and show a limited traceback.
        """
    try:
        recursionindex = traceback.recursionindex()
    except Exception as e:
        max_frames = 10
        extraline: Optional[str] = f'!!! Recursion error detected, but an error occurred locating the origin of recursion.\n  The following exception happened when comparing locals in the stack frame:\n    {type(e).__name__}: {e!s}\n  Displaying first and last {max_frames} stack frames out of {len(traceback)}.'
        traceback = traceback[:max_frames] + traceback[-max_frames:]
    else:
        if recursionindex is not None:
            extraline = '!!! Recursion detected (same locals & position)'
            traceback = traceback[:recursionindex + 1]
        else:
            extraline = None
    return (traceback, extraline)