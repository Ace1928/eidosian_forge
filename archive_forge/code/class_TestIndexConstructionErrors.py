from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
class TestIndexConstructionErrors:

    def test_constructor_overflow_int64(self):
        msg = 'The elements provided in the data cannot all be casted to the dtype int64'
        with pytest.raises(OverflowError, match=msg):
            Index([np.iinfo(np.uint64).max - 1], dtype='int64')