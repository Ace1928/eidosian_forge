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
def _parseSSL(factory, port, privateKey='server.pem', certKey=None, sslmethod=None, interface='', backlog=50, extraCertChain=None, dhParameters=None):
    """
    Internal parser function for L{_parseServer} to convert the string
    arguments for an SSL (over TCP/IPv4) stream endpoint into the structured
    arguments.

    @param factory: the protocol factory being parsed, or L{None}.  (This was a
        leftover argument from when this code was in C{strports}, and is now
        mostly None and unused.)
    @type factory: L{IProtocolFactory} or L{None}

    @param port: the integer port number to bind
    @type port: C{str}

    @param interface: the interface IP to listen on
    @param backlog: the length of the listen queue
    @type backlog: C{str}

    @param privateKey: The file name of a PEM format private key file.
    @type privateKey: C{str}

    @param certKey: The file name of a PEM format certificate file.
    @type certKey: C{str}

    @param sslmethod: The string name of an SSL method, based on the name of a
        constant in C{OpenSSL.SSL}.
    @type sslmethod: C{str}

    @param extraCertChain: The path of a file containing one or more
        certificates in PEM format that establish the chain from a root CA to
        the CA that signed your C{certKey}.
    @type extraCertChain: L{str}

    @param dhParameters: The file name of a file containing parameters that are
        required for Diffie-Hellman key exchange.  If this is not specified,
        the forward secret C{DHE} ciphers aren't available for servers.
    @type dhParameters: L{str}

    @return: a 2-tuple of (args, kwargs), describing  the parameters to
        L{IReactorSSL.listenSSL} (or, modulo argument 2, the factory, arguments
        to L{SSL4ServerEndpoint}.
    """
    from twisted.internet import ssl
    if certKey is None:
        certKey = privateKey
    kw = {}
    if sslmethod is not None:
        kw['method'] = getattr(ssl.SSL, sslmethod)
    certPEM = FilePath(certKey).getContent()
    keyPEM = FilePath(privateKey).getContent()
    privateCertificate = ssl.PrivateCertificate.loadPEM(certPEM + b'\n' + keyPEM)
    if extraCertChain is not None:
        matches = re.findall('(-----BEGIN CERTIFICATE-----\\n.+?\\n-----END CERTIFICATE-----)', nativeString(FilePath(extraCertChain).getContent()), flags=re.DOTALL)
        chainCertificates = [ssl.Certificate.loadPEM(chainCertPEM).original for chainCertPEM in matches]
        if not chainCertificates:
            raise ValueError("Specified chain file '%s' doesn't contain any valid certificates in PEM format." % (extraCertChain,))
    else:
        chainCertificates = None
    if dhParameters is not None:
        dhParameters = ssl.DiffieHellmanParameters.fromFile(FilePath(dhParameters))
    cf = ssl.CertificateOptions(privateKey=privateCertificate.privateKey.original, certificate=privateCertificate.original, extraCertChain=chainCertificates, dhParameters=dhParameters, **kw)
    return ((int(port), factory, cf), {'interface': interface, 'backlog': int(backlog)})