import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
def _read_until(sock, char):
    """
    Read from the socket until the character is received.
    """
    chunks = []
    while True:
        chunk = sock.recv(1)
        chunks.append(chunk)
        if chunk == char:
            break
    return b''.join(chunks)