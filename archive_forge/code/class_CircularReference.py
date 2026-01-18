from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class CircularReference(object):

    def create(self):
        x = {}
        x['y'] = x
        return x