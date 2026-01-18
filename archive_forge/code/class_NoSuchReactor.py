from typing import Iterable, cast
from zope.interface import Attribute, Interface, implementer
from twisted.internet.interfaces import IReactorCore
from twisted.plugin import IPlugin, getPlugins
from twisted.python.reflect import namedAny
class NoSuchReactor(KeyError):
    """
    Raised when an attempt is made to install a reactor which cannot be found.
    """