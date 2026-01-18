from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestLibMissing:

    @pytest.mark.parametrize('func', [libmissing.checknull, isna])
    @pytest.mark.parametrize('value', na_vals + sometimes_na_vals)
    def test_checknull_na_vals(self, func, value):
        assert func(value)

    @pytest.mark.parametrize('func', [libmissing.checknull, isna])
    @pytest.mark.parametrize('value', inf_vals)
    def test_checknull_inf_vals(self, func, value):
        assert not func(value)

    @pytest.mark.parametrize('func', [libmissing.checknull, isna])
    @pytest.mark.parametrize('value', int_na_vals)
    def test_checknull_intna_vals(self, func, value):
        assert not func(value)

    @pytest.mark.parametrize('func', [libmissing.checknull, isna])
    @pytest.mark.parametrize('value', never_na_vals)
    def test_checknull_never_na_vals(self, func, value):
        assert not func(value)

    @pytest.mark.parametrize('value', na_vals + sometimes_na_vals)
    def test_checknull_old_na_vals(self, value):
        assert libmissing.checknull(value, inf_as_na=True)

    @pytest.mark.parametrize('value', inf_vals)
    def test_checknull_old_inf_vals(self, value):
        assert libmissing.checknull(value, inf_as_na=True)

    @pytest.mark.parametrize('value', int_na_vals)
    def test_checknull_old_intna_vals(self, value):
        assert not libmissing.checknull(value, inf_as_na=True)

    @pytest.mark.parametrize('value', int_na_vals)
    def test_checknull_old_never_na_vals(self, value):
        assert not libmissing.checknull(value, inf_as_na=True)

    def test_is_matching_na(self, nulls_fixture, nulls_fixture2):
        left = nulls_fixture
        right = nulls_fixture2
        assert libmissing.is_matching_na(left, left)
        if left is right:
            assert libmissing.is_matching_na(left, right)
        elif is_float(left) and is_float(right):
            assert libmissing.is_matching_na(left, right)
        elif type(left) is type(right):
            assert libmissing.is_matching_na(left, right)
        else:
            assert not libmissing.is_matching_na(left, right)

    def test_is_matching_na_nan_matches_none(self):
        assert not libmissing.is_matching_na(None, np.nan)
        assert not libmissing.is_matching_na(np.nan, None)
        assert libmissing.is_matching_na(None, np.nan, nan_matches_none=True)
        assert libmissing.is_matching_na(np.nan, None, nan_matches_none=True)