import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
class UnsupportedInterfaces(StrcredException):
    """
    Raised when an application is given a checker to use that does not
    provide any of the application's supported credentials interfaces.
    """