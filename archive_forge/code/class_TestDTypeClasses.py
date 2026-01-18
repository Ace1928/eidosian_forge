import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
class TestDTypeClasses:

    @pytest.mark.parametrize('dtype', list(np.typecodes['All']) + [rational])
    def test_basic_dtypes_subclass_properties(self, dtype):
        dtype = np.dtype(dtype)
        assert isinstance(dtype, np.dtype)
        assert type(dtype) is not np.dtype
        if dtype.type.__name__ != 'rational':
            dt_name = type(dtype).__name__.lower().removesuffix('dtype')
            if dt_name == 'uint' or dt_name == 'int':
                dt_name += 'c'
            sc_name = dtype.type.__name__
            assert dt_name == sc_name.strip('_')
            assert type(dtype).__module__ == 'numpy.dtypes'
            assert getattr(numpy.dtypes, type(dtype).__name__) is type(dtype)
        else:
            assert type(dtype).__name__ == 'dtype[rational]'
            assert type(dtype).__module__ == 'numpy'
        assert not type(dtype)._abstract
        parametric = (np.void, np.str_, np.bytes_, np.datetime64, np.timedelta64)
        if dtype.type not in parametric:
            assert not type(dtype)._parametric
            assert type(dtype)() is dtype
        else:
            assert type(dtype)._parametric
            with assert_raises(TypeError):
                type(dtype)()

    def test_dtype_superclass(self):
        assert type(np.dtype) is not type
        assert isinstance(np.dtype, type)
        assert type(np.dtype).__name__ == '_DTypeMeta'
        assert type(np.dtype).__module__ == 'numpy'
        assert np.dtype._abstract

    def test_is_numeric(self):
        all_codes = set(np.typecodes['All'])
        numeric_codes = set(np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?')
        non_numeric_codes = all_codes - numeric_codes
        for code in numeric_codes:
            assert type(np.dtype(code))._is_numeric
        for code in non_numeric_codes:
            assert not type(np.dtype(code))._is_numeric

    @pytest.mark.parametrize('int_', ['UInt', 'Int'])
    @pytest.mark.parametrize('size', [8, 16, 32, 64])
    def test_integer_alias_names(self, int_, size):
        DType = getattr(numpy.dtypes, f'{int_}{size}DType')
        sctype = getattr(numpy, f'{int_.lower()}{size}')
        assert DType.type is sctype
        assert DType.__name__.lower().removesuffix('dtype') == sctype.__name__

    @pytest.mark.parametrize('name', ['Half', 'Float', 'Double', 'CFloat', 'CDouble'])
    def test_float_alias_names(self, name):
        with pytest.raises(AttributeError):
            getattr(numpy.dtypes, name + 'DType') is numpy.dtypes.Float16DType