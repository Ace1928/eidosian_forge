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
def _add_digest(self, digest, address, requests):
    address = (address[0], int(address[1]))
    msg = requests[address]
    msg.add_digest(digest)
    if msg.digest_count >= self.batch_size:
        try:
            return self.send(msg, address)
        finally:
            del requests[address]