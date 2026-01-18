from socket import (
from typing import (
from zope.interface import implementer
from twisted.internet._idna import _idnaBytes
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import (
from twisted.internet.threads import deferToThreadPool
from twisted.logger import Logger
from twisted.python.compat import nativeString
@implementer(IResolutionReceiver)
class FirstOneWins:
    """
    An L{IResolutionReceiver} which fires a L{Deferred} with its first result.
    """

    def __init__(self, deferred: 'Deferred[str]'):
        """
        @param deferred: The L{Deferred} to fire when the first resolution
            result arrives.
        """
        self._deferred = deferred
        self._resolved = False

    def resolutionBegan(self, resolution: IHostResolution) -> None:
        """
        See L{IResolutionReceiver.resolutionBegan}

        @param resolution: See L{IResolutionReceiver.resolutionBegan}
        """
        self._resolution = resolution

    def addressResolved(self, address: IAddress) -> None:
        """
        See L{IResolutionReceiver.addressResolved}

        @param address: See L{IResolutionReceiver.addressResolved}
        """
        if self._resolved:
            return
        self._resolved = True
        assert isinstance(address, IPv4Address)
        self._deferred.callback(address.host)

    def resolutionComplete(self) -> None:
        """
        See L{IResolutionReceiver.resolutionComplete}
        """
        if self._resolved:
            return
        self._deferred.errback(DNSLookupError(self._resolution.name))