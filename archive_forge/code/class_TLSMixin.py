from typing import Optional, Sequence, Type
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.endpoints import (
from twisted.internet.error import ConnectionClosed
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
class TLSMixin:
    requiredInterfaces: Optional[Sequence[Type[Interface]]] = [IReactorSSL]
    if platform.isWindows():
        msg = "For some reason, these reactors don't deal with SSL disconnection correctly on Windows.  See #3371."
        skippedReactors = {'twisted.internet.glib2reactor.Glib2Reactor': msg, 'twisted.internet.gtk2reactor.Gtk2Reactor': msg}