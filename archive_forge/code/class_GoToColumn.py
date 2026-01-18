import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class GoToColumn(_PositionToken):
    """Token to advance to a specific column of input text; useful for tabular report scraping."""

    def __init__(self, colno):
        super(GoToColumn, self).__init__()
        self.col = colno

    def preParse(self, instring, loc):
        if col(loc, instring) != self.col:
            instrlen = len(instring)
            if self.ignoreExprs:
                loc = self._skipIgnorables(instring, loc)
            while loc < instrlen and instring[loc].isspace() and (col(loc, instring) != self.col):
                loc += 1
        return loc

    def parseImpl(self, instring, loc, doActions=True):
        thiscol = col(loc, instring)
        if thiscol > self.col:
            raise ParseException(instring, loc, 'Text not in expected column', self)
        newloc = loc + self.col - thiscol
        ret = instring[loc:newloc]
        return (newloc, ret)