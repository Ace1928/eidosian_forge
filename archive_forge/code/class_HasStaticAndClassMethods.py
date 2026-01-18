from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class HasStaticAndClassMethods(object):
    """A class with a static method and a class method."""
    CLASS_STATE = 1

    def __init__(self, instance_state):
        self.instance_state = instance_state

    @staticmethod
    def static_fn(args):
        return args

    @classmethod
    def class_fn(cls, args):
        return args + cls.CLASS_STATE