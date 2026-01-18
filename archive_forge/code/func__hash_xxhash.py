from __future__ import annotations
import binascii
import hashlib
def _hash_xxhash(buf):
    """
        Produce a 8-bytes hash of *buf* using xxHash.
        """
    return xxhash.xxh64(buf).digest()