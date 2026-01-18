import logging
import os
class LazyHostName:
    """Avoid importing socket and calling gethostname() unnecessarily"""

    def __str__(self):
        import socket
        return socket.gethostname()