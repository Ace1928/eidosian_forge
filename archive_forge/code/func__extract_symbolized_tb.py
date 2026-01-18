from types import TracebackType
from typing import List, Optional
import tempfile
import traceback
import contextlib
import inspect
import os.path
def _extract_symbolized_tb(tb, skip):
    """
    Given a symbolized traceback from symbolize_tracebacks, return a StackSummary object of
    pre-processed stack trace entries.
    """
    stack = traceback.StackSummary()
    for f in reversed(tb[skip:]):
        stack.append(traceback.FrameSummary(f['filename'], f['line'], f['name']))
    return stack