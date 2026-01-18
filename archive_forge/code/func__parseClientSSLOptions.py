import os
import re
import socket
import warnings
from typing import Optional, Sequence, Type
from unicodedata import normalize
from zope.interface import directlyProvides, implementer, provider
from constantly import NamedConstant, Names
from incremental import Version
from twisted.internet import defer, error, fdesc, interfaces, threads
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.address import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, ProcessProtocol, Protocol
from twisted.internet._resolver import HostResolution
from twisted.internet.defer import Deferred
from twisted.internet.task import LoopingCall
from twisted.logger import Logger
from twisted.plugin import IPlugin, getPlugins
from twisted.python import deprecate, log
from twisted.python.compat import _matchingString, iterbytes, nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.systemd import ListenFDs
from ._idna import _idnaBytes, _idnaText
def _parseClientSSLOptions(kwargs):
    """
    Parse common arguments for SSL endpoints, creating an L{CertificateOptions}
    instance.

    @param kwargs: A dict of keyword arguments to be parsed, potentially
        containing keys C{certKey}, C{privateKey}, C{caCertsDir}, and
        C{hostname}.  See L{_parseClientSSL}.
    @type kwargs: L{dict}

    @return: The remaining arguments, including a new key C{sslContextFactory}.
    """
    hostname = kwargs.pop('hostname', None)
    clientCertificate = _privateCertFromPaths(kwargs.pop('certKey', None), kwargs.pop('privateKey', None))
    trustRoot = _parseTrustRootPath(kwargs.pop('caCertsDir', None))
    if hostname is not None:
        configuration = optionsForClientTLS(_idnaText(hostname), trustRoot=trustRoot, clientCertificate=clientCertificate)
    else:
        if clientCertificate is not None:
            privateKeyOpenSSL = clientCertificate.privateKey.original
            certificateOpenSSL = clientCertificate.original
        else:
            privateKeyOpenSSL = None
            certificateOpenSSL = None
        configuration = CertificateOptions(trustRoot=trustRoot, privateKey=privateKeyOpenSSL, certificate=certificateOpenSSL)
    kwargs['sslContextFactory'] = configuration
    return kwargs