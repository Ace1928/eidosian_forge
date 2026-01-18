import operator
import sys
import types
import unittest
import abc
import pytest
import six
class myiter(six.Iterator):

    def __next__(self):
        return 13