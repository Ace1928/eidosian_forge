import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
class TestCategoricalReprWithFactor:

    def test_print(self, using_infer_string):
        factor = Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'], ordered=True)
        if using_infer_string:
            expected = ["['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']", 'Categories (3, string): [a < b < c]']
        else:
            expected = ["['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']", "Categories (3, object): ['a' < 'b' < 'c']"]
        expected = '\n'.join(expected)
        actual = repr(factor)
        assert actual == expected