from __future__ import annotations
import warnings
from typing import Any
class ConstantWarning(DeprecationWarning):
    """Accessing a constant no longer in current CODATA data set"""
    pass