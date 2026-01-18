from collections import OrderedDict
import functools
import itertools
import operator
import re
import sys
from pyparsing import (
import numpy
class ExpressionError(SyntaxError):
    """A Snuggs-specific syntax error."""
    filename = '<string>'
    lineno = 1