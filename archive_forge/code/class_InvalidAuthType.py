import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
class InvalidAuthType(StrcredException):
    """
    Raised when a user provides an invalid identifier for the
    authentication plugin (known as the authType).
    """