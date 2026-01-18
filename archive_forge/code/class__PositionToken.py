import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class _PositionToken(Token):

    def __init__(self):
        super(_PositionToken, self).__init__()
        self.name = self.__class__.__name__
        self.mayReturnEmpty = True
        self.mayIndexError = False