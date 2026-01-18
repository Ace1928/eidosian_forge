import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _encrypt_on(sock, context, hostname):
    """Wrap a socket in SSL/TLS. Arguments:
        - sock: Socket to wrap
        - context: SSL context to use for the encrypted connection
        Returns:
        - sock: New, encrypted socket.
        """
    if context is None:
        context = ssl._create_stdlib_context()
    return context.wrap_socket(sock, server_hostname=hostname)