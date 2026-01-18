from collections import deque
import os
import typing
from typing import (
from abc import ABC, abstractmethod
from enum import Enum
import string
import copy
import warnings
import re
import sys
from collections.abc import Iterable
import traceback
import types
from operator import itemgetter
from functools import wraps
from threading import RLock
from pathlib import Path
from .util import (
from .exceptions import *
from .actions import *
from .results import ParseResults, _ParseResultsWithOffset
from .unicode import pyparsing_unicode
class AtLineStart(ParseElementEnhance):
    """Matches if an expression matches at the beginning of a line within
    the parse string

    Example::

        test = '''\\
        AAA this line
        AAA and this line
          AAA but not this one
        B AAA and definitely not this one
        '''

        for t in (AtLineStart('AAA') + rest_of_line).search_string(test):
            print(t)

    prints::

        ['AAA', ' this line']
        ['AAA', ' and this line']

    """

    def __init__(self, expr: Union[ParserElement, str]):
        super().__init__(expr)
        self.callPreparse = False

    def parseImpl(self, instring, loc, doActions=True):
        if col(loc, instring) != 1:
            raise ParseException(instring, loc, 'not found at line start')
        return super().parseImpl(instring, loc, doActions)