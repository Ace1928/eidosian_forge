from __future__ import annotations
from contextlib import (
import inspect
import re
import sys
from typing import (
import warnings
from pandas.compat import PY311
def _is_unexpected_warning(actual_warning: warnings.WarningMessage, expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None) -> bool:
    """Check if the actual warning issued is unexpected."""
    if actual_warning and (not expected_warning):
        return True
    expected_warning = cast(type[Warning], expected_warning)
    return bool(not issubclass(actual_warning.category, expected_warning))