from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class ArgumentSerializer(object):
    """Base class for generating string representations of a flag value."""

    def serialize(self, value):
        """Returns a serialized string of the value."""
        return _helpers.str_or_unicode(value)