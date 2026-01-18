import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
class TestAssertNoGcCycles:
    """ Test assert_no_gc_cycles """

    def test_passes(self):

        def no_cycle():
            b = []
            b.append([])
            return b
        with assert_no_gc_cycles():
            no_cycle()
        assert_no_gc_cycles(no_cycle)

    def test_asserts(self):

        def make_cycle():
            a = []
            a.append(a)
            a.append(a)
            return a
        with assert_raises(AssertionError):
            with assert_no_gc_cycles():
                make_cycle()
        with assert_raises(AssertionError):
            assert_no_gc_cycles(make_cycle)

    @pytest.mark.slow
    def test_fails(self):
        """
        Test that in cases where the garbage cannot be collected, we raise an
        error, instead of hanging forever trying to clear it.
        """

        class ReferenceCycleInDel:
            """
            An object that not only contains a reference cycle, but creates new
            cycles whenever it's garbage-collected and its __del__ runs
            """
            make_cycle = True

            def __init__(self):
                self.cycle = self

            def __del__(self):
                self.cycle = None
                if ReferenceCycleInDel.make_cycle:
                    ReferenceCycleInDel()
        try:
            w = weakref.ref(ReferenceCycleInDel())
            try:
                with assert_raises(RuntimeError):
                    assert_no_gc_cycles(lambda: None)
            except AssertionError:
                if w() is not None:
                    pytest.skip('GC does not call __del__ on cyclic objects')
                    raise
        finally:
            ReferenceCycleInDel.make_cycle = False