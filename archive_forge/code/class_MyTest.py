import operator
import sys
import types
import unittest
import abc
import pytest
import six
@six.python_2_unicode_compatible
class MyTest(object):

    def __str__(self):
        return six.u('hello')

    def __bytes__(self):
        return six.b('hello')