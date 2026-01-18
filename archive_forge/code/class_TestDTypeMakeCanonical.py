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
class TestDTypeMakeCanonical:

    def check_canonical(self, dtype, canonical):
        """
        Check most properties relevant to "canonical" versions of a dtype,
        which is mainly native byte order for datatypes supporting this.

        The main work is checking structured dtypes with fields, where we
        reproduce most the actual logic used in the C-code.
        """
        assert type(dtype) is type(canonical)
        assert np.can_cast(dtype, canonical, casting='equiv')
        assert np.can_cast(canonical, dtype, casting='equiv')
        assert canonical.isnative
        assert np.result_type(canonical) == canonical
        if not dtype.names:
            assert dtype.flags == canonical.flags
            return
        assert dtype.flags & 16
        assert dtype.fields.keys() == canonical.fields.keys()

        def aligned_offset(offset, alignment):
            return -(-offset // alignment) * alignment
        totalsize = 0
        max_alignment = 1
        for name in dtype.names:
            new_field_descr = canonical.fields[name][0]
            self.check_canonical(dtype.fields[name][0], new_field_descr)
            expected = 27 & new_field_descr.flags
            assert canonical.flags & expected == expected
            if canonical.isalignedstruct:
                totalsize = aligned_offset(totalsize, new_field_descr.alignment)
                max_alignment = max(new_field_descr.alignment, max_alignment)
            assert canonical.fields[name][1] == totalsize
            assert dtype.fields[name][2:] == canonical.fields[name][2:]
            totalsize += new_field_descr.itemsize
        if canonical.isalignedstruct:
            totalsize = aligned_offset(totalsize, max_alignment)
        assert canonical.itemsize == totalsize
        assert canonical.alignment == max_alignment

    def test_simple(self):
        dt = np.dtype('>i4')
        assert np.result_type(dt).isnative
        assert np.result_type(dt).num == dt.num
        struct_dt = np.dtype('>i4,<i1,i8,V3')[['f0', 'f2']]
        canonical = np.result_type(struct_dt)
        assert canonical.itemsize == 4 + 8
        assert canonical.isnative
        struct_dt = np.dtype('>i1,<i4,i8,V3', align=True)[['f0', 'f2']]
        canonical = np.result_type(struct_dt)
        assert canonical.isalignedstruct
        assert canonical.itemsize == np.dtype('i8').alignment + 8
        assert canonical.isnative

    def test_object_flag_not_inherited(self):
        arr = np.ones(3, 'i,O,i')[['f0', 'f2']]
        assert arr.dtype.hasobject
        canonical_dt = np.result_type(arr.dtype)
        assert not canonical_dt.hasobject

    @pytest.mark.slow
    @hypothesis.given(dtype=hynp.nested_dtypes())
    def test_make_canonical_hypothesis(self, dtype):
        canonical = np.result_type(dtype)
        self.check_canonical(dtype, canonical)
        two_arg_result = np.result_type(dtype, dtype)
        assert np.can_cast(two_arg_result, canonical, casting='no')

    @pytest.mark.slow
    @hypothesis.given(dtype=hypothesis.extra.numpy.array_dtypes(subtype_strategy=hypothesis.extra.numpy.array_dtypes(), min_size=5, max_size=10, allow_subarrays=True))
    def test_structured(self, dtype):
        field_subset = random.sample(dtype.names, k=4)
        dtype_with_empty_space = dtype[field_subset]
        assert dtype_with_empty_space.itemsize == dtype.itemsize
        canonicalized = np.result_type(dtype_with_empty_space)
        self.check_canonical(dtype_with_empty_space, canonicalized)
        two_arg_result = np.promote_types(dtype_with_empty_space, dtype_with_empty_space)
        assert np.can_cast(two_arg_result, canonicalized, casting='no')
        dtype_aligned = np.dtype(dtype.descr, align=not dtype.isalignedstruct)
        dtype_with_empty_space = dtype_aligned[field_subset]
        assert dtype_with_empty_space.itemsize == dtype_aligned.itemsize
        canonicalized = np.result_type(dtype_with_empty_space)
        self.check_canonical(dtype_with_empty_space, canonicalized)
        two_arg_result = np.promote_types(dtype_with_empty_space, dtype_with_empty_space)
        assert np.can_cast(two_arg_result, canonicalized, casting='no')