from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class SubPoint(NamedTuplePoint):
    """Used for verifying subclasses of namedtuples behave as intended."""

    def coordinate_sum(self):
        return self.x + self.y