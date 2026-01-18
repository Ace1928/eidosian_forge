from __future__ import annotations
import base64
import hashlib
import secrets
import sys
def accept_key(key: str) -> str:
    """
    Compute the value of the Sec-WebSocket-Accept header.

    Args:
        key: value of the Sec-WebSocket-Key header.

    """
    sha1 = hashlib.sha1((key + GUID).encode()).digest()
    return base64.b64encode(sha1).decode()