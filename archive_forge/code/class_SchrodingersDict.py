import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
class SchrodingersDict(dict):

    def __getattribute__(inner_self, attr):
        self.fail('doing attribute lookup might have side effects')