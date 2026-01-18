import sys
import threading
import time
import traceback
from types import SimpleNamespace
def _excepthook(self, args, use_thread_hook):
    recursionLimit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(recursionLimit + 100)
        global callbacks, clear_tracebacks
        header = '===== %s =====' % str(time.strftime('%Y.%m.%d %H:%m:%S', time.localtime(time.time())))
        try:
            print(header)
        except Exception:
            sys.stderr.write('Warning: stdout is broken! Falling back to stderr.\n')
            sys.stdout = sys.stderr
        if use_thread_hook:
            ret = self.orig_threading_excepthook(args)
        else:
            ret = self.orig_sys_excepthook(args.exc_type, args.exc_value, args.exc_traceback)
        for cb in callbacks:
            try:
                cb(args)
            except Exception:
                print('   --------------------------------------------------------------')
                print('      Error occurred during exception callback %s' % str(cb))
                print('   --------------------------------------------------------------')
                traceback.print_exception(*sys.exc_info())
        for cb in old_callbacks:
            try:
                cb(args.exc_type, args.exc_value, args.exc_traceback)
            except Exception:
                print('   --------------------------------------------------------------')
                print('      Error occurred during exception callback %s' % str(cb))
                print('   --------------------------------------------------------------')
                traceback.print_exception(*sys.exc_info())
        if clear_tracebacks is True:
            sys.last_traceback = None
        return ret
    finally:
        sys.setrecursionlimit(recursionLimit)