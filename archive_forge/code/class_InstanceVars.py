from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class InstanceVars(object):

    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def run(self, arg1, arg2):
        return (self.arg1, self.arg2, arg1, arg2)