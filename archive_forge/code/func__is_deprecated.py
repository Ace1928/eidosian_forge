import functools
import warnings
def _is_deprecated(func):
    """Helper to check if func is wrapped by our deprecated decorator"""
    closures = getattr(func, '__closure__', [])
    if closures is None:
        closures = []
    is_deprecated = 'deprecated' in ''.join([c.cell_contents for c in closures if isinstance(c.cell_contents, str)])
    return is_deprecated