import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
class OpAssoc(Enum):
    """Enumeration of operator associativity
    - used in constructing InfixNotationOperatorSpec for :class:`infix_notation`"""
    LEFT = 1
    RIGHT = 2