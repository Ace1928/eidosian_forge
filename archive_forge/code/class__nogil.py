from __future__ import absolute_import
import math, sys
class _nogil(object):
    """Support for 'with nogil' statement and @nogil decorator.
    """

    def __call__(self, x):
        if callable(x):
            return x
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_class, exc, tb):
        return exc_class is None