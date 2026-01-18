import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class StringEnd(_PositionToken):
    """Matches if current position is at the end of the parse string"""

    def __init__(self):
        super(StringEnd, self).__init__()
        self.errmsg = 'Expected end of text'

    def parseImpl(self, instring, loc, doActions=True):
        if loc < len(instring):
            raise ParseException(instring, loc, self.errmsg, self)
        elif loc == len(instring):
            return (loc + 1, [])
        elif loc > len(instring):
            return (loc, [])
        else:
            raise ParseException(instring, loc, self.errmsg, self)