from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
@property
def _label_macro(self) -> str:
    """Label macro, extracted from self.label, like \\label{ref}."""
    return f'\\label{{{self.label}}}' if self.label else ''