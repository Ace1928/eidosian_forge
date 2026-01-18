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
class ScrapyClientContextFactory(BrowserLikePolicyForHTTPS):
    """
    Non-peer-certificate verifying HTTPS context factory

    Default OpenSSL method is TLS_METHOD (also called SSLv23_METHOD)
    which allows TLS protocol negotiation

    'A TLS/SSL connection established with [this method] may
     understand the TLSv1, TLSv1.1 and TLSv1.2 protocols.'
    """

    def __init__(self, method: int=SSL.SSLv23_METHOD, tls_verbose_logging: bool=False, tls_ciphers: Optional[str]=None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._ssl_method: int = method
        self.tls_verbose_logging: bool = tls_verbose_logging
        self.tls_ciphers: AcceptableCiphers
        if tls_ciphers:
            self.tls_ciphers = AcceptableCiphers.fromOpenSSLCipherString(tls_ciphers)
        else:
            self.tls_ciphers = DEFAULT_CIPHERS

    @classmethod
    def from_settings(cls, settings: BaseSettings, method: int=SSL.SSLv23_METHOD, *args: Any, **kwargs: Any):
        tls_verbose_logging: bool = settings.getbool('DOWNLOADER_CLIENT_TLS_VERBOSE_LOGGING')
        tls_ciphers: Optional[str] = settings['DOWNLOADER_CLIENT_TLS_CIPHERS']
        return cls(*args, method=method, tls_verbose_logging=tls_verbose_logging, tls_ciphers=tls_ciphers, **kwargs)

    def getCertificateOptions(self) -> CertificateOptions:
        return CertificateOptions(verify=False, method=getattr(self, 'method', getattr(self, '_ssl_method', None)), fixBrokenPeers=True, acceptableCiphers=self.tls_ciphers)

    def getContext(self, hostname: Any=None, port: Any=None) -> SSL.Context:
        ctx = self.getCertificateOptions().getContext()
        ctx.set_options(4)
        return ctx

    def creatorForNetloc(self, hostname: bytes, port: int) -> 'ClientTLSOptions':
        return ScrapyClientTLSOptions(hostname.decode('ascii'), self.getContext(), verbose_logging=self.tls_verbose_logging)