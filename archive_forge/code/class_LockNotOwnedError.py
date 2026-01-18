import asyncio
import builtins
class LockNotOwnedError(LockError):
    """Error trying to extend or release a lock that is (no longer) owned"""
    pass