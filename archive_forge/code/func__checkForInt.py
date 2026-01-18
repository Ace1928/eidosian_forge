import doctest
import collections
def _checkForInt(arg):
    """Raises an exception if arg is not an int. Always returns None."""
    if not isinstance(arg, int):
        raise PyRectException('argument must be int or float, not %s' % arg.__class__.__name__)