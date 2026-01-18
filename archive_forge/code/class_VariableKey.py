import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
class VariableKey:
    """
    A dictionary key which is a variable.

    @ivar item: The variable AST object.
    """

    def __init__(self, item):
        self.name = item.id

    def __eq__(self, compare):
        return compare.__class__ == self.__class__ and compare.name == self.name

    def __hash__(self):
        return hash(self.name)