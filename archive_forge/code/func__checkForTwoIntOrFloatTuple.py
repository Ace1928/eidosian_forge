import doctest
import collections
def _checkForTwoIntOrFloatTuple(arg):
    try:
        if not isinstance(arg[0], (int, float)) or not isinstance(arg[1], (int, float)):
            raise PyRectException('argument must be a two-item tuple containing int or float values')
    except:
        raise PyRectException('argument must be a two-item tuple containing int or float values')