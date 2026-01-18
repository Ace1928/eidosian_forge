from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
def _insert_index(self, data):
    data = data.copy()
    idx_nlevels = data.index.nlevels
    if idx_nlevels == 1:
        data.insert(0, 'Index', data.index)
    else:
        for i in range(idx_nlevels):
            data.insert(i, f'Index{i}', data.index._get_level_values(i))
    col_nlevels = data.columns.nlevels
    if col_nlevels > 1:
        col = data.columns._get_level_values(0)
        values = [data.columns._get_level_values(i)._values for i in range(1, col_nlevels)]
        col_df = pd.DataFrame(values)
        data.columns = col_df.columns
        data = pd.concat([col_df, data])
        data.columns = col
    return data