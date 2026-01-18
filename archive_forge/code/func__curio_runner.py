import ast
import asyncio
import inspect
from functools import wraps
def _curio_runner(coroutine):
    """
    handler for curio autoawait
    """
    import curio
    return curio.run(coroutine)