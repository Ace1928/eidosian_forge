import errno
import select
from zope.interface import Attribute, Interface, declarations, implementer
from twisted.internet import main, posixbase
from twisted.internet.interfaces import IReactorDaemonize, IReactorFDSet
from twisted.python import failure, log
class _IKQueue(Interface):
    """
    An interface for KQueue implementations.
    """
    kqueue = Attribute('An implementation of kqueue(2).')
    kevent = Attribute('An implementation of kevent(2).')