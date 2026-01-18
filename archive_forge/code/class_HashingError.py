from __future__ import annotations
class HashingError(Argon2Error):
    """
    Raised if hashing failed.

    You can find the original error message from Argon2 in ``args[0]``.
    """