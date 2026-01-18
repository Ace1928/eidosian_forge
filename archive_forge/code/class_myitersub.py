import operator
import sys
import types
import unittest
import abc
import pytest
import six
class myitersub(myiter):

    def __next__(self):
        return 14