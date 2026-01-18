from collections import OrderedDict, deque
from datetime import date, time, datetime
from decimal import Decimal
from fractions import Fraction
import ast
import enum
import typing
class CannotEval(Exception):

    def __repr__(self):
        return self.__class__.__name__
    __str__ = __repr__