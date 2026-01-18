from typing import final
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_numeric_dtype
class BaseBooleanReduceTests(BaseReduceTests):

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name not in ['any', 'all']:
            pytest.skip('These are tested in BaseNumericReduceTests')
        return True