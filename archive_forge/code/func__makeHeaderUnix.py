from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def _makeHeaderUnix(sig: bytes=V2_SIGNATURE, verCom: bytes=b'!', famProto: bytes=b'1', addrLength: bytes=b'\x00\xd8', addrs: bytes=(b'/home/tests/mysockets/sock' + b'\x00' * 82) * 2) -> bytes:
    """
    Construct a version 2 IPv4 header with custom bytes.

    @param sig: The protocol signature; defaults to valid L{V2_SIGNATURE}.
    @type sig: L{bytes}

    @param verCom: Protocol version and command.  Defaults to V2 PROXY.
    @type verCom: L{bytes}

    @param famProto: Address family and protocol.  Defaults to AF_UNIX/STREAM.
    @type famProto: L{bytes}

    @param addrLength: Network-endian byte length of payload.  Defaults to 108
        bytes for 2 null terminated paths.
    @type addrLength: L{bytes}

    @param addrs: Address payload.  Defaults to C{/home/tests/mysockets/sock}
        for source and destination paths.
    @type addrs: L{bytes}

    @return: A packet with header, addresses, and8 ports.
    @rtype: L{bytes}
    """
    return sig + verCom + famProto + addrLength + addrs