import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
@pytest.mark.parametrize('false_or_none', [False, None])
class TestFalseOrNoneExpectedWarning:

    def test_raise_on_warning(self, false_or_none):
        msg = 'Caused unexpected warning\\(s\\)'
        with pytest.raises(AssertionError, match=msg):
            with tm.assert_produces_warning(false_or_none):
                f()

    def test_no_raise_without_warning(self, false_or_none):
        with tm.assert_produces_warning(false_or_none):
            pass

    def test_no_raise_with_false_raise_on_extra(self, false_or_none):
        with tm.assert_produces_warning(false_or_none, raise_on_extra_warnings=False):
            f()