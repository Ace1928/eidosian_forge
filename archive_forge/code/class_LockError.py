import asyncio
import builtins
class LockError(RedisError, ValueError):
    """Errors acquiring or releasing a lock"""
    pass