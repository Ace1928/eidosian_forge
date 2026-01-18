from __future__ import annotations
import binascii
import hashlib
def _hash_sha1(buf):
    """
    Produce a 20-bytes hash of *buf* using SHA1.
    """
    return hashlib.sha1(buf).digest()