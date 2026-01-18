from __future__ import unicode_literals
class WinRMOperationTimeoutError(Exception):
    """
    Raised when a WinRM-level operation timeout (not a connection-level timeout) has occurred. This is
    considered a normal error that should be retried transparently by the client when waiting for output from
    a long-running process.
    """
    code = 500