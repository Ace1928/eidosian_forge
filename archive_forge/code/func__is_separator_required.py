from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _is_separator_required(self) -> bool:
    return bool(self.header and self.env_body)