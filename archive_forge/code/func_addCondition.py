import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
def addCondition(self, *fns, **kwargs):
    """Add a boolean predicate function to expression's list of parse actions. See 
        L{I{setParseAction}<setParseAction>} for function call signatures. Unlike C{setParseAction}, 
        functions passed to C{addCondition} need to return boolean success/fail of the condition.

        Optional keyword arguments:
         - message = define a custom message to be used in the raised exception
         - fatal   = if True, will raise ParseFatalException to stop parsing immediately; otherwise will raise ParseException
         
        Example::
            integer = Word(nums).setParseAction(lambda toks: int(toks[0]))
            year_int = integer.copy()
            year_int.addCondition(lambda toks: toks[0] >= 2000, message="Only support years 2000 and later")
            date_str = year_int + '/' + integer + '/' + integer

            result = date_str.parseString("1999/12/31")  # -> Exception: Only support years 2000 and later (at char 0), (line:1, col:1)
        """
    msg = kwargs.get('message', 'failed user-defined condition')
    exc_type = ParseFatalException if kwargs.get('fatal', False) else ParseException
    for fn in fns:

        def pa(s, l, t):
            if not bool(_trim_arity(fn)(s, l, t)):
                raise exc_type(s, l, msg)
        self.parseAction.append(pa)
    self.callDuringTry = self.callDuringTry or kwargs.get('callDuringTry', False)
    return self