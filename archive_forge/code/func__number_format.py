from __future__ import annotations
from collections.abc import (
import csv as csvlib
import os
from typing import (
import numpy as np
from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core.indexes.api import Index
from pandas.io.common import get_handle
@property
def _number_format(self) -> dict[str, Any]:
    """Dictionary used for storing number formatting settings."""
    return {'na_rep': self.na_rep, 'float_format': self.float_format, 'date_format': self.date_format, 'quoting': self.quoting, 'decimal': self.decimal}