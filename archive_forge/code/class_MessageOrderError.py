import socket
class MessageOrderError(SSHException):
    """
    Out-of-order protocol messages were received, violating "strict kex" mode.

    .. versionadded:: 3.4
    """
    pass