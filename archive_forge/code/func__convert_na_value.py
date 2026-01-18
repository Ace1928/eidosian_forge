import numpy as np
import pandas as pd
def _convert_na_value(ser, expected):
    if ser.dtype != object:
        if ser.dtype.storage == 'pyarrow_numpy':
            expected = expected.fillna(np.nan)
        else:
            expected = expected.fillna(pd.NA)
    return expected