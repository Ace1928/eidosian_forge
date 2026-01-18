import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
class SchrodingersCatsDict(dict):

    def __getattr__(inner_self, attr):
        self.fail('doing attribute lookup might have side effects')