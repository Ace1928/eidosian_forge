from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from six.moves import zip_longest
@classmethod
def _CmpHelper(cls, x, y):
    """Just a helper equivalent to the cmp() function in Python 2."""
    return (x > y) - (x < y)