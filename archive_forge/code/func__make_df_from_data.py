import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def _make_df_from_data(data):
    rows = {}
    for date in data:
        for level in data[date]:
            rows[date, level[0]] = {'A': level[1], 'B': level[2]}
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.names = ('Date', 'Item')
    return df