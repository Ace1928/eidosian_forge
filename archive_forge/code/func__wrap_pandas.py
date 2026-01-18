import numpy as np
import pandas as pd
from scipy import stats
def _wrap_pandas(self, value, name=None, columns=None):
    if not self._use_pandas:
        return value
    if value.ndim == 1:
        return pd.Series(value, index=self._row_labels, name=name)
    return pd.DataFrame(value, index=self._row_labels, columns=columns)