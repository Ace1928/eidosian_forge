from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
class PandasUtils(PandasLikeUtils[pd.DataFrame, pd.Series]):
    """A collection of pandas utils"""

    def concat_dfs(self, *dfs: DataFrame) -> DataFrame:
        return pd.concat(list(dfs))