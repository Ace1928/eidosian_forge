from __future__ import annotations
from contextlib import (
import inspect
import re
import sys
from typing import (
import warnings
from pandas.compat import PY311
def _assert_raised_with_correct_stacklevel(actual_warning: warnings.WarningMessage) -> None:
    frame = inspect.currentframe()
    for _ in range(4):
        frame = frame.f_back
    try:
        caller_filename = inspect.getfile(frame)
    finally:
        del frame
    msg = f'Warning not set with correct stacklevel. File where warning is raised: {actual_warning.filename} != {caller_filename}. Warning message: {actual_warning.message}'
    assert actual_warning.filename == caller_filename, msg