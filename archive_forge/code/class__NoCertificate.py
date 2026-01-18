from __future__ import annotations
import datetime
import decimal
import warnings
from functools import partial
from io import BytesIO
from itertools import count
from struct import pack
from types import MethodType
from typing import (
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, fail, maybeDeferred
from twisted.internet.error import ConnectionClosed, ConnectionLost, PeerVerifyError
from twisted.internet.interfaces import IFileDescriptorReceiver
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.basic import Int16StringReceiver, StatefulStringProtocol
from twisted.python import filepath, log
from twisted.python._tzhelper import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.reflect import accumulateClassDict
class _NoCertificate:
    """
    This is for peers which don't want to use a local certificate.  Used by
    AMP because AMP's internal language is all about certificates and this
    duck-types in the appropriate place; this API isn't really stable though,
    so it's not exposed anywhere public.

    For clients, it will use ephemeral DH keys, or whatever the default is for
    certificate-less clients in OpenSSL.  For servers, it will generate a
    temporary self-signed certificate with garbage values in the DN and use
    that.
    """

    def __init__(self, client):
        """
        Create a _NoCertificate which either is or isn't for the client side of
        the connection.

        @param client: True if we are a client and should truly have no
        certificate and be anonymous, False if we are a server and actually
        have to generate a temporary certificate.

        @type client: bool
        """
        self.client = client

    def options(self, *authorities):
        """
        Behaves like L{twisted.internet.ssl.PrivateCertificate.options}().
        """
        if not self.client:
            sharedDN = DN(CN='TEMPORARY CERTIFICATE')
            key = KeyPair.generate()
            cr = key.certificateRequest(sharedDN)
            sscrd = key.signCertificateRequest(sharedDN, cr, lambda dn: True, 1)
            cert = key.newCertificate(sscrd)
            return cert.options(*authorities)
        options = dict()
        if authorities:
            options.update(dict(verify=True, requireCertificate=True, caCerts=[auth.original for auth in authorities]))
        occo = CertificateOptions(**options)
        return occo