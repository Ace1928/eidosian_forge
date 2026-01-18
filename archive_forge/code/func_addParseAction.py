import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def addParseAction(self, *fns, **kwargs):
    """Add parse action to expression's list of parse actions. See L{I{setParseAction}<setParseAction>}."""
    self.parseAction += list(map(_trim_arity, list(fns)))
    self.callDuringTry = self.callDuringTry or ('callDuringTry' in kwargs and kwargs['callDuringTry'])
    return self