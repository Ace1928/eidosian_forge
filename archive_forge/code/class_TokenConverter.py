import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class TokenConverter(ParseElementEnhance):
    """Abstract subclass of C{ParseExpression}, for converting parsed results."""

    def __init__(self, expr, savelist=False):
        super(TokenConverter, self).__init__(expr)
        self.saveAsList = False