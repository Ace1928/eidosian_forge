from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class NonComparable(object):

    def __eq__(self, other):
        raise ValueError('Instances of this class cannot be compared.')

    def __ne__(self, other):
        raise ValueError('Instances of this class cannot be compared.')