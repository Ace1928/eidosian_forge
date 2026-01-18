from __future__ import annotations
from datetime import time
import math
from typing import TYPE_CHECKING
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import BaseExcelReader

            converts the contents of the cell into a pandas appropriate object
            