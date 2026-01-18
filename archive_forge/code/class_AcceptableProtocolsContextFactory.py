import warnings
from typing import TYPE_CHECKING, Any, List, Optional
from OpenSSL import SSL
from twisted.internet._sslverify import _setAcceptableProtocols
from twisted.internet.ssl import (
from twisted.web.client import BrowserLikePolicyForHTTPS
from twisted.web.iweb import IPolicyForHTTPS
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from scrapy.core.downloader.tls import (
from scrapy.settings import BaseSettings
from scrapy.utils.misc import create_instance, load_object
@implementer(IPolicyForHTTPS)
class AcceptableProtocolsContextFactory:
    """Context factory to used to override the acceptable protocols
    to set up the [OpenSSL.SSL.Context] for doing NPN and/or ALPN
    negotiation.
    """

    def __init__(self, context_factory: Any, acceptable_protocols: List[bytes]):
        verifyObject(IPolicyForHTTPS, context_factory)
        self._wrapped_context_factory: Any = context_factory
        self._acceptable_protocols: List[bytes] = acceptable_protocols

    def creatorForNetloc(self, hostname: bytes, port: int) -> 'ClientTLSOptions':
        options: 'ClientTLSOptions' = self._wrapped_context_factory.creatorForNetloc(hostname, port)
        _setAcceptableProtocols(options._ctx, self._acceptable_protocols)
        return options