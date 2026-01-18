import hashlib
import io
import json
import os
import platform
import random
import socket
import ssl
import threading
import time
import urllib.parse
from typing import (
import filelock
import urllib3
from blobfile import _xml as xml
def _read_with_deadline(fp: Any, sock: socket.socket, nbytes_to_read: int, deadline: float) -> bytes:
    timeout = deadline - time.time()
    if timeout <= 0:
        raise Timeout('Did not read enough bytes before deadline expired')
    sock.settimeout(timeout)
    try:
        initial_data = fp.peek()
    except socket.timeout:
        raise Timeout('Timed out waiting for read')
    buf = bytearray(nbytes_to_read)
    mv = memoryview(buf)
    mv[:len(initial_data)] = initial_data
    nbytes_read = len(initial_data)
    while nbytes_read < nbytes_to_read:
        timeout = deadline - time.time()
        if timeout <= 0:
            raise Timeout('Did not read enough bytes before deadline expired')
        sock.settimeout(timeout)
        try:
            n = sock.recv_into(mv[nbytes_read:])
        except socket.timeout:
            raise Timeout('Timed out waiting for read')
        nbytes_read += n
    return bytes(buf)