from __future__ import annotations
import codecs
import io
from typing import (
import warnings
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
from pandas.io.xml import (
@final
def _process_dataframe(self) -> dict[int | str, dict[str, Any]]:
    """
        Adjust Data Frame to fit xml output.

        This method will adjust underlying data frame for xml output,
        including optionally replacing missing values and including indexes.
        """
    df = self.frame
    if self.index:
        df = df.reset_index()
    if self.na_rep is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Downcasting object dtype arrays', category=FutureWarning)
            df = df.fillna(self.na_rep)
    return df.to_dict(orient='index')