from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class VarArgs(object):
    """Test class for testing Python Fire with a property with varargs."""

    def cumsums(self, *items):
        total = None
        sums = []
        for item in items:
            if total is None:
                total = item
            else:
                total += item
            sums.append(total)
        return sums

    def varchars(self, alpha=0, beta=0, *chars):
        return (alpha, beta, ''.join(chars))