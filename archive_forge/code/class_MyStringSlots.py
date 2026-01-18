import operator
import sys
import types
import unittest
import abc
import pytest
import six
class MyStringSlots(object):
    __slots__ = 'ab'