import os
import sys
import time
import warnings
import contextlib
import portend
@contextlib.contextmanager
def _safe_wait(host, port):
    """
    On systems where a loopback interface is not available and the
    server is bound to all interfaces, it's difficult to determine
    whether the server is in fact occupying the port. In this case,
    just issue a warning and move on. See issue #1100.
    """
    try:
        yield
    except portend.Timeout:
        if host == portend.client_host(host):
            raise
        msg = 'Unable to verify that the server is bound on %r' % port
        warnings.warn(msg)