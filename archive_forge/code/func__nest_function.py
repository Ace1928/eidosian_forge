import ast
import sys
import importlib.util
def _nest_function(ob, func_name, lineno, end_lineno, is_async=False):
    """Return a Function after nesting within ob."""
    return Function(ob.module, func_name, ob.file, lineno, parent=ob, is_async=is_async, end_lineno=end_lineno)