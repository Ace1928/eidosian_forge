from __future__ import annotations
from typing import (
import numpy as np
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
import pandas as pd
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import BaseExcelReader
def _get_cell_value(self, cell) -> Scalar | NaTType:
    from odf.namespaces import OFFICENS
    if str(cell) == '#N/A':
        return np.nan
    cell_type = cell.attributes.get((OFFICENS, 'value-type'))
    if cell_type == 'boolean':
        if str(cell) == 'TRUE':
            return True
        return False
    if cell_type is None:
        return self.empty_value
    elif cell_type == 'float':
        cell_value = float(cell.attributes.get((OFFICENS, 'value')))
        val = int(cell_value)
        if val == cell_value:
            return val
        return cell_value
    elif cell_type == 'percentage':
        cell_value = cell.attributes.get((OFFICENS, 'value'))
        return float(cell_value)
    elif cell_type == 'string':
        return self._get_cell_string_value(cell)
    elif cell_type == 'currency':
        cell_value = cell.attributes.get((OFFICENS, 'value'))
        return float(cell_value)
    elif cell_type == 'date':
        cell_value = cell.attributes.get((OFFICENS, 'date-value'))
        return pd.Timestamp(cell_value)
    elif cell_type == 'time':
        stamp = pd.Timestamp(str(cell))
        return cast(Scalar, stamp.time())
    else:
        self.close()
        raise ValueError(f'Unrecognized type {cell_type}')