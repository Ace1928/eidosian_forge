import eventlet
from eventlet.green import thread
from eventlet.green import time
from eventlet.support import greenlets as greenlet
def _fixup_thread(t):
    global __threading
    if not __threading:
        __threading = __import__('threading')
    if hasattr(__threading.Thread, 'get_name') and (not hasattr(t, 'get_name')):
        t.get_name = t.getName
    return t