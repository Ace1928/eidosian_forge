import itertools
import os
import sys
import tempfile
import unittest
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arrays_equal
from ..array_sequence import ArraySequence, concatenate, is_array_sequence
def _test_unary(op, arrseq):
    orig = arrseq.copy()
    seq = getattr(orig, op)()
    assert seq is not orig
    check_arr_seq(seq, [getattr(d, op)() for d in orig])