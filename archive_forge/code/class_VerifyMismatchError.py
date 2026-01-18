from __future__ import annotations
class VerifyMismatchError(VerificationError):
    """
    The secret does not match the hash.

    Subclass of :exc:`argon2.exceptions.VerificationError`.

    .. versionadded:: 16.1.0
    """