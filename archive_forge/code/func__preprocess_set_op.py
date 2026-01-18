from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
def _preprocess_set_op(self, ndf1: T, ndf2: T) -> Tuple[T, T]:
    assert_or_throw(len(list(ndf1.columns)) == len(list(ndf2.columns)), ValueError('two dataframes have different number of columns'))
    ndf2.columns = ndf1.columns
    return (ndf1, ndf2)