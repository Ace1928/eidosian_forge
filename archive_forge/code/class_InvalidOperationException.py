import collections
import collections.abc
import operator
import warnings
class InvalidOperationException(Exception):
    """Raised when trying to use Policy class as a dict."""
    pass