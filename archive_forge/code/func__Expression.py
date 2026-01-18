from __future__ import print_function, unicode_literals
import six
import sys
import ast
import os
import tokenize
from six import StringIO
def _Expression(self, tree):
    self.dispatch(tree.body)