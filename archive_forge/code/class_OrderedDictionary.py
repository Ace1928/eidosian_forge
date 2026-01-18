from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class OrderedDictionary(object):

    def empty(self):
        return collections.OrderedDict()

    def non_empty(self):
        ordered_dict = collections.OrderedDict()
        ordered_dict['A'] = 'A'
        ordered_dict[2] = 2
        return ordered_dict