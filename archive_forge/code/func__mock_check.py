import time
import email
import socket
import logging
import functools
import collections
import pyzor.digest
import pyzor.account
import pyzor.message
import pyzor.hacks.py26
def _mock_check(self, digests, address=None):
    msg = b'Code: %s\nDiag: OK\nPV: %s\nThread: 1024\nCount: 0\nWL-Count: 0' % (pyzor.message.Response.ok_code, pyzor.proto_version)
    return email.message_from_bytes(msg, _class=pyzor.message.Response)