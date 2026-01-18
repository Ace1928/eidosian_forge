import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def _parseCache(self, instring, loc, doActions=True, callPreParse=True):
    lookup = (self, instring, loc, callPreParse, doActions)
    if lookup in ParserElement._exprArgCache:
        value = ParserElement._exprArgCache[lookup]
        if isinstance(value, Exception):
            raise value
        return (value[0], value[1].copy())
    else:
        try:
            value = self._parseNoCache(instring, loc, doActions, callPreParse)
            ParserElement._exprArgCache[lookup] = (value[0], value[1].copy())
            return value
        except ParseBaseException as pe:
            pe.__traceback__ = None
            ParserElement._exprArgCache[lookup] = pe
            raise