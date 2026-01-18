import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
class StrcredException(Exception):
    """
    Base exception class for strcred.
    """