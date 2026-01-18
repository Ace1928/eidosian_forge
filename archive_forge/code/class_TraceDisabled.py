import struct
from oslo_log import log as logging
class TraceDisabled(object):
    """A logger-like thing that swallows tracing when we do not want it."""

    def debug(self, *a, **k):
        pass
    info = debug
    warning = debug
    error = debug