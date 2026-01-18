from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
class SetitemCastingEquivalents:
    """
    Check each of several methods that _should_ be equivalent to `obj[key] = val`

    We assume that
        - obj.index is the default Index(range(len(obj)))
        - the setitem does not expand the obj
    """

    @pytest.fixture
    def is_inplace(self, obj, expected):
        """
        Whether we expect the setting to be in-place or not.
        """
        return expected.dtype == obj.dtype

    def check_indexer(self, obj, key, expected, val, indexer, is_inplace):
        orig = obj
        obj = obj.copy()
        arr = obj._values
        indexer(obj)[key] = val
        tm.assert_series_equal(obj, expected)
        self._check_inplace(is_inplace, orig, arr, obj)

    def _check_inplace(self, is_inplace, orig, arr, obj):
        if is_inplace is None:
            pass
        elif is_inplace:
            if arr.dtype.kind in ['m', 'M']:
                assert arr._ndarray is obj._values._ndarray
            else:
                assert obj._values is arr
        else:
            tm.assert_equal(arr, orig._values)

    def test_int_key(self, obj, key, expected, warn, val, indexer_sli, is_inplace):
        if not isinstance(key, int):
            pytest.skip('Not relevant for int key')
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            self.check_indexer(obj, key, expected, val, indexer_sli, is_inplace)
        if indexer_sli is tm.loc:
            with tm.assert_produces_warning(warn, match='incompatible dtype'):
                self.check_indexer(obj, key, expected, val, tm.at, is_inplace)
        elif indexer_sli is tm.iloc:
            with tm.assert_produces_warning(warn, match='incompatible dtype'):
                self.check_indexer(obj, key, expected, val, tm.iat, is_inplace)
        rng = range(key, key + 1)
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            self.check_indexer(obj, rng, expected, val, indexer_sli, is_inplace)
        if indexer_sli is not tm.loc:
            slc = slice(key, key + 1)
            with tm.assert_produces_warning(warn, match='incompatible dtype'):
                self.check_indexer(obj, slc, expected, val, indexer_sli, is_inplace)
        ilkey = [key]
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)
        indkey = np.array(ilkey)
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)
        genkey = (x for x in [key])
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)

    def test_slice_key(self, obj, key, expected, warn, val, indexer_sli, is_inplace):
        if not isinstance(key, slice):
            pytest.skip('Not relevant for slice key')
        if indexer_sli is not tm.loc:
            with tm.assert_produces_warning(warn, match='incompatible dtype'):
                self.check_indexer(obj, key, expected, val, indexer_sli, is_inplace)
        ilkey = list(range(len(obj)))[key]
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)
        indkey = np.array(ilkey)
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)
        genkey = (x for x in indkey)
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)

    def test_mask_key(self, obj, key, expected, warn, val, indexer_sli):
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        obj = obj.copy()
        if is_list_like(val) and len(val) < mask.sum():
            msg = 'boolean index did not match indexed array along dimension'
            with pytest.raises(IndexError, match=msg):
                indexer_sli(obj)[mask] = val
            return
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            indexer_sli(obj)[mask] = val
        tm.assert_series_equal(obj, expected)

    def test_series_where(self, obj, key, expected, warn, val, is_inplace):
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        if is_list_like(val) and len(val) < len(obj):
            msg = 'operands could not be broadcast together with shapes'
            with pytest.raises(ValueError, match=msg):
                obj.where(~mask, val)
            return
        orig = obj
        obj = obj.copy()
        arr = obj._values
        res = obj.where(~mask, val)
        if val is NA and res.dtype == object:
            expected = expected.fillna(NA)
        elif val is None and res.dtype == object:
            assert expected.dtype == object
            expected = expected.copy()
            expected[expected.isna()] = None
        tm.assert_series_equal(res, expected)
        self._check_inplace(is_inplace, orig, arr, obj)

    def test_index_where(self, obj, key, expected, warn, val, using_infer_string):
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        if using_infer_string and obj.dtype == object:
            with pytest.raises(TypeError, match='Scalar must'):
                Index(obj).where(~mask, val)
        else:
            res = Index(obj).where(~mask, val)
            expected_idx = Index(expected, dtype=expected.dtype)
            tm.assert_index_equal(res, expected_idx)

    def test_index_putmask(self, obj, key, expected, warn, val, using_infer_string):
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        if using_infer_string and obj.dtype == object:
            with pytest.raises(TypeError, match='Scalar must'):
                Index(obj).putmask(mask, val)
        else:
            res = Index(obj).putmask(mask, val)
            tm.assert_index_equal(res, Index(expected, dtype=expected.dtype))