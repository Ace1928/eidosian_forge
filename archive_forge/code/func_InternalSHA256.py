import hashlib
import socket
import time
from pyu2f import errors
from pyu2f import hardware
from pyu2f import hidtransport
from pyu2f import model
def InternalSHA256(self, string):
    md = hashlib.sha256()
    md.update(string.encode())
    return md.digest()