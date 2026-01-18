import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
class NoSuchMethod(AttributeError):
    """Raised if there is no such remote method"""