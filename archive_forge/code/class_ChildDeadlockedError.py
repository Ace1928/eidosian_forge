import asyncio
import builtins
class ChildDeadlockedError(Exception):
    """Error indicating that a child process is deadlocked after a fork()"""
    pass