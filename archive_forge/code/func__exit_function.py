import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def _exit_function(info=info, debug=debug, _run_finalizers=_run_finalizers, active_children=process.active_children, current_process=process.current_process):
    global _exiting
    if not _exiting:
        _exiting = True
        info('process shutting down')
        debug('running all "atexit" finalizers with priority >= 0')
        _run_finalizers(0)
        if current_process() is not None:
            for p in active_children():
                if p.daemon:
                    info('calling terminate() for daemon %s', p.name)
                    p._popen.terminate()
            for p in active_children():
                info('calling join() for process %s', p.name)
                p.join()
        debug('running the remaining "atexit" finalizers')
        _run_finalizers()