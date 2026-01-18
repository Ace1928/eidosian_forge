import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
class TestRoundTrips:

    def test_pickle_roundtrip(self, index):
        result = tm.round_trip_pickle(index)
        tm.assert_index_equal(result, index, exact=True)
        if result.nlevels > 1:
            assert index.equal_levels(result)

    def test_pickle_preserves_name(self, index):
        original_name, index.name = (index.name, 'foo')
        unpickled = tm.round_trip_pickle(index)
        assert index.equals(unpickled)
        index.name = original_name