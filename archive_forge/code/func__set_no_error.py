from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def _set_no_error():
    spglib_error.message = 'no error'