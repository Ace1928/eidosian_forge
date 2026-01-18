import operator
import sys
import types
import unittest
import abc
import pytest
import six
class MySlots(object):
    __slots__ = ['a', 'b']