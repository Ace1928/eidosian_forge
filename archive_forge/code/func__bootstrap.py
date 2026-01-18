import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
def _bootstrap(self, parent_sentinel=None):
    from . import util, context
    global _current_process, _parent_process, _process_counter, _children
    try:
        if self._start_method is not None:
            context._force_start_method(self._start_method)
        _process_counter = itertools.count(1)
        _children = set()
        util._close_stdin()
        old_process = _current_process
        _current_process = self
        _parent_process = _ParentProcess(self._parent_name, self._parent_pid, parent_sentinel)
        if threading._HAVE_THREAD_NATIVE_ID:
            threading.main_thread()._set_native_id()
        try:
            self._after_fork()
        finally:
            del old_process
        util.info('child process calling self.run()')
        try:
            self.run()
            exitcode = 0
        finally:
            util._exit_function()
    except SystemExit as e:
        if e.code is None:
            exitcode = 0
        elif isinstance(e.code, int):
            exitcode = e.code
        else:
            sys.stderr.write(str(e.code) + '\n')
            exitcode = 1
    except:
        exitcode = 1
        import traceback
        sys.stderr.write('Process %s:\n' % self.name)
        traceback.print_exc()
    finally:
        threading._shutdown()
        util.info('process exiting with exitcode %d' % exitcode)
        util._flush_std_streams()
    return exitcode