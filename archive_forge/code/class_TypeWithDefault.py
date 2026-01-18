from __future__ import absolute_import
import urllib3
import copy
import logging
import multiprocessing
import sys
from six import iteritems
from six import with_metaclass
from six.moves import http_client as httplib
class TypeWithDefault(type):

    def __init__(cls, name, bases, dct):
        super(TypeWithDefault, cls).__init__(name, bases, dct)
        cls._default = None

    def __call__(cls):
        if cls._default == None:
            cls._default = type.__call__(cls)
        return copy.copy(cls._default)

    def set_default(cls, default):
        cls._default = copy.copy(default)