import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def _test_from_int_memoryview():
    a = array.array('i', range(3, 103))
    mv = memoryview(a)
    vec = ri.IntSexpVector.from_memoryview(mv)
    assert tuple(range(3, 103)) == tuple(vec)