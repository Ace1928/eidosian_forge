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
Context factory to used to override the acceptable protocols
    to set up the [OpenSSL.SSL.Context] for doing NPN and/or ALPN
    negotiation.
    