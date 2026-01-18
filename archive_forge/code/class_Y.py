import operator
import sys
import types
import unittest
import abc
import pytest
import six
class Y(six.with_metaclass(MetaSub, X)):
    pass