import atexit
import os
import signal
import sys
import ovs.vlog
def _call_hooks(signr):
    global recurse
    if recurse:
        return
    recurse = True
    for hook, cancel, run_at_exit in _hooks:
        if signr != 0 or run_at_exit:
            hook()