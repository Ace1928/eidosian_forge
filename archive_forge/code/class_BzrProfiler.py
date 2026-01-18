import _thread
import codecs
import operator
import os
import pickle
import sys
import threading
from typing import Dict, TextIO
from _lsprof import Profiler, profiler_entry
from . import errors
class BzrProfiler:
    """Bzr utility wrapper around Profiler.

    For most uses the module level 'profile()' function will be suitable.
    However profiling when a simple wrapped function isn't available may
    be easier to accomplish using this class.

    To use it, create a BzrProfiler and call start() on it. Some arbitrary
    time later call stop() to stop profiling and retrieve the statistics
    from the code executed in the interim.

    Note that profiling involves a threading.Lock around the actual profiling.
    This is needed because profiling involves global manipulation of the python
    interpreter state. As such you cannot perform multiple profiles at once.
    Trying to do so will lock out the second profiler unless the global
    breezy.lsprof.BzrProfiler.profiler_block is set to 0. Setting it to 0 will
    cause profiling to fail rather than blocking.
    """
    profiler_block = 1
    'Serialise rather than failing to profile concurrent profile requests.'
    profiler_lock = threading.Lock()
    'Global lock used to serialise profiles.'

    def start(self):
        """Start profiling.

        This hooks into threading and will record all calls made until
        stop() is called.
        """
        self._g_threadmap = {}
        self.p = Profiler()
        permitted = self.__class__.profiler_lock.acquire(self.__class__.profiler_block)
        if not permitted:
            raise errors.InternalBzrError(msg='Already profiling something')
        try:
            self.p.enable(subcalls=True)
            threading.setprofile(self._thread_profile)
        except BaseException:
            self.__class__.profiler_lock.release()
            raise

    def stop(self):
        """Stop profiling.

        This unhooks from threading and cleans up the profiler, returning
        the gathered Stats object.

        :return: A breezy.lsprof.Stats object.
        """
        try:
            self.p.disable()
            for pp in self._g_threadmap.values():
                pp.disable()
            threading.setprofile(None)
            p = self.p
            self.p = None
            threads = {}
            for tid, pp in self._g_threadmap.items():
                threads[tid] = Stats(pp.getstats(), {})
            self._g_threadmap = None
            return Stats(p.getstats(), threads)
        finally:
            self.__class__.profiler_lock.release()

    def _thread_profile(self, f, *args, **kwds):
        thr = _thread.get_ident()
        self._g_threadmap[thr] = p = Profiler()
        p.enable(subcalls=True, builtins=True)