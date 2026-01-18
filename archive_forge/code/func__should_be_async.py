import ast
import asyncio
import inspect
from functools import wraps
def _should_be_async(cell: str) -> bool:
    """Detect if a block of code need to be wrapped in an `async def`

    Attempt to parse the block of code, it it compile we're fine.
    Otherwise we  wrap if and try to compile.

    If it works, assume it should be async. Otherwise Return False.

    Not handled yet: If the block of code has a return statement as the top
    level, it will be seen as async. This is a know limitation.
    """
    try:
        code = compile(cell, '<>', 'exec', flags=getattr(ast, 'PyCF_ALLOW_TOP_LEVEL_AWAIT', 0))
        return inspect.CO_COROUTINE & code.co_flags == inspect.CO_COROUTINE
    except (SyntaxError, MemoryError):
        return False