import json
import re
import warnings
import numpy as np
import pandas as pd
import pandas.io.formats.format as fmt
def _format_column(x):
    dtype_kind = x.dtype.kind
    if dtype_kind in ['b', 'i', 's']:
        return x
    try:
        x = fmt.format_array(x._values, None, justify='all', leading_space=False)
    except TypeError:
        x = fmt.format_array(x._values, None, justify='all')
    if dtype_kind == 'f':
        try:
            return np.array(x).astype(float)
        except ValueError:
            pass
    return x