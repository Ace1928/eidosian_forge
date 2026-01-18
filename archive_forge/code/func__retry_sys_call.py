from zmq.error import InterruptedSystemCall, _check_rc, _check_version
from ._cffi import ffi
from ._cffi import lib as C
def _retry_sys_call(f, *args, **kwargs):
    """make a call, retrying if interrupted with EINTR"""
    while True:
        rc = f(*args)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break