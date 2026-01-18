from __future__ import annotations
from contextlib import (
import inspect
import re
import sys
from typing import (
import warnings
from pandas.compat import PY311
def _assert_caught_expected_warning(*, caught_warnings: Sequence[warnings.WarningMessage], expected_warning: type[Warning], match: str | None, check_stacklevel: bool) -> None:
    """Assert that there was the expected warning among the caught warnings."""
    saw_warning = False
    matched_message = False
    unmatched_messages = []
    for actual_warning in caught_warnings:
        if issubclass(actual_warning.category, expected_warning):
            saw_warning = True
            if check_stacklevel:
                _assert_raised_with_correct_stacklevel(actual_warning)
            if match is not None:
                if re.search(match, str(actual_warning.message)):
                    matched_message = True
                else:
                    unmatched_messages.append(actual_warning.message)
    if not saw_warning:
        raise AssertionError(f'Did not see expected warning of class {repr(expected_warning.__name__)}')
    if match and (not matched_message):
        raise AssertionError(f"Did not see warning {repr(expected_warning.__name__)} matching '{match}'. The emitted warning messages are {unmatched_messages}")