import doctest
import collections
def _checkForIntOrFloat(arg):
    """Raises an exception if arg is not an int or float. Always returns None."""
    if not isinstance(arg, (int, float)):
        raise PyRectException('argument must be int or float, not %s' % arg.__class__.__name__)