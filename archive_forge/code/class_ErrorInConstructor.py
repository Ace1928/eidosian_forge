from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class ErrorInConstructor(object):

    def __init__(self, value='value'):
        self.value = value
        raise ValueError('Error in constructor')