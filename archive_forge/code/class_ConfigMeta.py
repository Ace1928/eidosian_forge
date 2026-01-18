from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
class ConfigMeta(type):
    """Metaclass that calls __set_name__ on all subclass descriptors for pythons
  prior to 3.6
  """

    def __new__(mcs, name, bases, dct):
        subclass = type.__new__(mcs, name, bases, dct)
        if sys.version_info < (3, 6, 0):
            _emulate_setname(subclass)
        return subclass

    def __setattr__(cls, name, value):
        if hasattr(value, '__set_name__'):
            getattr(value, '__set_name__')(cls, name)