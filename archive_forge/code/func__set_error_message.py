from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def _set_error_message():
    spglib_error.message = _spglib.error_message()