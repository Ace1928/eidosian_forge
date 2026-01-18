from typing import Optional, Union
from twisted.internet import interfaces
from twisted.internet.endpoints import _WrapperServerEndpoint
from twisted.protocols import policies
from . import _info
from ._exceptions import InvalidProxyHeader
from ._v1parser import V1Parser
from ._v2parser import V2Parser
class HAProxyProtocolWrapper(policies.ProtocolWrapper):
    """
    A Protocol wrapper that provides HAProxy support.

    This protocol reads the PROXY stream header, v1 or v2, parses the provided
    connection data, and modifies the behavior of getPeer and getHost to return
    the data provided by the PROXY header.
    """

    def __init__(self, factory: policies.WrappingFactory, wrappedProtocol: interfaces.IProtocol):
        super().__init__(factory, wrappedProtocol)
        self._proxyInfo: Optional[_info.ProxyInfo] = None
        self._parser: Union[V2Parser, V1Parser, None] = None

    def dataReceived(self, data: bytes) -> None:
        if self._proxyInfo is not None:
            return self.wrappedProtocol.dataReceived(data)
        parser = self._parser
        if parser is None:
            if len(data) >= 16 and data[:12] == V2Parser.PREFIX and (ord(data[12:13]) & 240 == 32):
                self._parser = parser = V2Parser()
            elif len(data) >= 8 and data[:5] == V1Parser.PROXYSTR:
                self._parser = parser = V1Parser()
            else:
                self.loseConnection()
                return None
        try:
            self._proxyInfo, remaining = parser.feed(data)
            if remaining:
                self.wrappedProtocol.dataReceived(remaining)
        except InvalidProxyHeader:
            self.loseConnection()

    def getPeer(self) -> interfaces.IAddress:
        if self._proxyInfo and self._proxyInfo.source:
            return self._proxyInfo.source
        assert self.transport
        return self.transport.getPeer()

    def getHost(self) -> interfaces.IAddress:
        if self._proxyInfo and self._proxyInfo.destination:
            return self._proxyInfo.destination
        assert self.transport
        return self.transport.getHost()