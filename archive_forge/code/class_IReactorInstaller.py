from typing import Iterable, cast
from zope.interface import Attribute, Interface, implementer
from twisted.internet.interfaces import IReactorCore
from twisted.plugin import IPlugin, getPlugins
from twisted.python.reflect import namedAny
class IReactorInstaller(Interface):
    """
    Definition of a reactor which can probably be installed.
    """
    shortName = Attribute('\n    A brief string giving the user-facing name of this reactor.\n    ')
    description = Attribute('\n    A longer string giving a user-facing description of this reactor.\n    ')

    def install() -> None:
        """
        Install this reactor.
        """