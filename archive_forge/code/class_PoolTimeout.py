from __future__ import annotations
import contextlib
import typing
class PoolTimeout(TimeoutException):
    """
    Timed out waiting to acquire a connection from the pool.
    """