from __future__ import annotations
from typing import Collection
class KeyAlreadyPresent(TOMLKitError):
    """
    An already present key was used.
    """

    def __init__(self, key):
        key = getattr(key, 'key', key)
        message = f'Key "{key}" already exists.'
        super().__init__(message)