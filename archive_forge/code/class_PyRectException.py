import doctest
import collections
class PyRectException(Exception):
    """
    This class exists for PyRect exceptions. If the PyRect module raises any
    non-PyRectException exceptions, this indicates there's a bug in PyRect.
    """
    pass