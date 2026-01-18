import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
class TestBzrProfiler(tests.TestCase):
    _test_needs_features = [features.lsprof_feature]

    def test_start_call_stuff_stop(self):
        profiler = lsprof.BzrProfiler()
        profiler.start()
        try:

            def a_function():
                pass
            a_function()
        finally:
            stats = profiler.stop()
        stats.freeze()
        lines = [str(data) for data in stats.data]
        lines = [line for line in lines if 'a_function' in line]
        self.assertLength(1, lines)

    def test_block_0(self):
        self.overrideAttr(lsprof.BzrProfiler, 'profiler_block', 0)
        inner_calls = []

        def inner():
            profiler = lsprof.BzrProfiler()
            self.assertRaises(errors.BzrError, profiler.start)
            inner_calls.append(True)
        lsprof.profile(inner)
        self.assertLength(1, inner_calls)

    def test_block_1(self):
        calls = []

        def profiled():
            calls.append('profiled')

        def do_profile():
            lsprof.profile(profiled)
            calls.append('after_profiled')
        thread = threading.Thread(target=do_profile)
        lsprof.BzrProfiler.profiler_lock.acquire()
        try:
            try:
                thread.start()
            finally:
                lsprof.BzrProfiler.profiler_lock.release()
        finally:
            thread.join()
        self.assertLength(2, calls)