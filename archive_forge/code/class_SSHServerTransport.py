from __future__ import annotations
import binascii
import hmac
import struct
import types
import zlib
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Any, Callable, Dict, Tuple, Union
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import dh, ec, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from typing_extensions import Literal
from twisted import __version__ as twisted_version
from twisted.conch.ssh import _kex, address, keys
from twisted.conch.ssh.common import MP, NS, ffs, getMP, getNS
from twisted.internet import defer, protocol
from twisted.logger import Logger
from twisted.python import randbytes
from twisted.python.compat import iterbytes, networkString
class SSHServerTransport(SSHTransportBase):
    """
    SSHServerTransport implements the server side of the SSH protocol.

    @ivar isClient: since we are never the client, this is always False.

    @ivar ignoreNextPacket: if True, ignore the next key exchange packet.  This
        is set when the client sends a guessed key exchange packet but with
        an incorrect guess.

    @ivar dhGexRequest: the KEX_DH_GEX_REQUEST(_OLD) that the client sent.
        The key generation needs this to be stored.

    @ivar g: the Diffie-Hellman group generator.

    @ivar p: the Diffie-Hellman group prime.
    """
    isClient = False
    ignoreNextPacket = 0

    def _getHostKeys(self, keyAlg):
        """
        Get the public and private host keys corresponding to the given
        public key signature algorithm.

        The factory stores public and private host keys by their key format,
        which is not quite the same as the key signature algorithm: for
        example, an ssh-rsa key can sign using any of the ssh-rsa,
        rsa-sha2-256, or rsa-sha2-512 algorithms.

        @type keyAlg: L{bytes}
        @param keyAlg: A public key signature algorithm name.

        @rtype: 2-L{tuple} of L{keys.Key}
        @return: The public and private host keys.

        @raises KeyError: if the factory does not have both a public and a
        private host key for this signature algorithm.
        """
        if keyAlg in {b'rsa-sha2-256', b'rsa-sha2-512'}:
            keyFormat = b'ssh-rsa'
        else:
            keyFormat = keyAlg
        return (self.factory.publicKeys[keyFormat], self.factory.privateKeys[keyFormat])

    def ssh_KEXINIT(self, packet):
        """
        Called when we receive a MSG_KEXINIT message.  For a description
        of the packet, see SSHTransportBase.ssh_KEXINIT().  Additionally,
        this method checks if a guessed key exchange packet was sent.  If
        it was sent, and it guessed incorrectly, the next key exchange
        packet MUST be ignored.
        """
        retval = SSHTransportBase.ssh_KEXINIT(self, packet)
        if not retval:
            return
        else:
            kexAlgs, keyAlgs, rest = retval
        if ord(rest[0:1]):
            if kexAlgs[0] != self.supportedKeyExchanges[0] or keyAlgs[0] != self.supportedPublicKeys[0]:
                self.ignoreNextPacket = True

    def _ssh_KEX_ECDH_INIT(self, packet):
        """
        Called from L{ssh_KEX_DH_GEX_REQUEST_OLD} to handle
        elliptic curve key exchanges.

        Payload::

            string client Elliptic Curve Diffie-Hellman public key

        Just like L{_ssh_KEXDH_INIT} this message type is also not dispatched
        directly. Extra check to determine if this is really KEX_ECDH_INIT
        is required.

        First we load the host's public/private keys.
        Then we generate the ECDH public/private keypair for the given curve.
        With that we generate the shared secret key.
        Then we compute the hash to sign and send back to the client
        Along with the server's public key and the ECDH public key.

        @type packet: L{bytes}
        @param packet: The message data.

        @return: None.
        """
        pktPub, packet = getNS(packet)
        pubHostKey, privHostKey = self._getHostKeys(self.keyAlg)
        ecPriv = self._generateECPrivateKey()
        self.ecPub = ecPriv.public_key()
        encPub = self._encodeECPublicKey(self.ecPub)
        sharedSecret = self._generateECSharedSecret(ecPriv, pktPub)
        h = _kex.getHashProcessor(self.kexAlg)()
        h.update(NS(self.otherVersionString))
        h.update(NS(self.ourVersionString))
        h.update(NS(self.otherKexInitPayload))
        h.update(NS(self.ourKexInitPayload))
        h.update(NS(pubHostKey.blob()))
        h.update(NS(pktPub))
        h.update(NS(encPub))
        h.update(sharedSecret)
        exchangeHash = h.digest()
        self.sendPacket(MSG_KEXDH_REPLY, NS(pubHostKey.blob()) + NS(encPub) + NS(privHostKey.sign(exchangeHash, signatureType=self.keyAlg)))
        self._keySetup(sharedSecret, exchangeHash)

    def _ssh_KEXDH_INIT(self, packet):
        """
        Called to handle the beginning of a non-group key exchange.

        Unlike other message types, this is not dispatched automatically.  It
        is called from C{ssh_KEX_DH_GEX_REQUEST_OLD} because an extra check is
        required to determine if this is really a KEXDH_INIT message or if it
        is a KEX_DH_GEX_REQUEST_OLD message.

        The KEXDH_INIT payload::

                integer e (the client's Diffie-Hellman public key)

        We send the KEXDH_REPLY with our host key and signature.

        @type packet: L{bytes}
        @param packet: The message data.
        """
        clientDHpublicKey, foo = getMP(packet)
        pubHostKey, privHostKey = self._getHostKeys(self.keyAlg)
        self.g, self.p = _kex.getDHGeneratorAndPrime(self.kexAlg)
        self._startEphemeralDH()
        sharedSecret = self._finishEphemeralDH(clientDHpublicKey)
        h = _kex.getHashProcessor(self.kexAlg)()
        h.update(NS(self.otherVersionString))
        h.update(NS(self.ourVersionString))
        h.update(NS(self.otherKexInitPayload))
        h.update(NS(self.ourKexInitPayload))
        h.update(NS(pubHostKey.blob()))
        h.update(MP(clientDHpublicKey))
        h.update(self.dhSecretKeyPublicMP)
        h.update(sharedSecret)
        exchangeHash = h.digest()
        self.sendPacket(MSG_KEXDH_REPLY, NS(pubHostKey.blob()) + self.dhSecretKeyPublicMP + NS(privHostKey.sign(exchangeHash, signatureType=self.keyAlg)))
        self._keySetup(sharedSecret, exchangeHash)

    def ssh_KEX_DH_GEX_REQUEST_OLD(self, packet):
        """
        This represents different key exchange methods that share the same
        integer value.  If the message is determined to be a KEXDH_INIT,
        L{_ssh_KEXDH_INIT} is called to handle it. If it is a KEX_ECDH_INIT,
        L{_ssh_KEX_ECDH_INIT} is called.
        Otherwise, for KEX_DH_GEX_REQUEST_OLD payload::

                integer ideal (ideal size for the Diffie-Hellman prime)

            We send the KEX_DH_GEX_GROUP message with the group that is
            closest in size to ideal.

        If we were told to ignore the next key exchange packet by ssh_KEXINIT,
        drop it on the floor and return.

        @type packet: L{bytes}
        @param packet: The message data.
        """
        if self.ignoreNextPacket:
            self.ignoreNextPacket = 0
            return
        if _kex.isFixedGroup(self.kexAlg):
            return self._ssh_KEXDH_INIT(packet)
        elif _kex.isEllipticCurve(self.kexAlg):
            return self._ssh_KEX_ECDH_INIT(packet)
        else:
            self.dhGexRequest = packet
            ideal = struct.unpack('>L', packet)[0]
            self.g, self.p = self.factory.getDHPrime(ideal)
            self._startEphemeralDH()
            self.sendPacket(MSG_KEX_DH_GEX_GROUP, MP(self.p) + MP(self.g))

    def ssh_KEX_DH_GEX_REQUEST(self, packet):
        """
        Called when we receive a MSG_KEX_DH_GEX_REQUEST message.  Payload::
            integer minimum
            integer ideal
            integer maximum

        The client is asking for a Diffie-Hellman group between minimum and
        maximum size, and close to ideal if possible.  We reply with a
        MSG_KEX_DH_GEX_GROUP message.

        If we were told to ignore the next key exchange packet by ssh_KEXINIT,
        drop it on the floor and return.

        @type packet: L{bytes}
        @param packet: The message data.
        """
        if self.ignoreNextPacket:
            self.ignoreNextPacket = 0
            return
        self.dhGexRequest = packet
        min, ideal, max = struct.unpack('>3L', packet)
        self.g, self.p = self.factory.getDHPrime(ideal)
        self._startEphemeralDH()
        self.sendPacket(MSG_KEX_DH_GEX_GROUP, MP(self.p) + MP(self.g))

    def ssh_KEX_DH_GEX_INIT(self, packet):
        """
        Called when we get a MSG_KEX_DH_GEX_INIT message.  Payload::
            integer e (client DH public key)

        We send the MSG_KEX_DH_GEX_REPLY message with our host key and
        signature.

        @type packet: L{bytes}
        @param packet: The message data.
        """
        clientDHpublicKey, foo = getMP(packet)
        pubHostKey, privHostKey = self._getHostKeys(self.keyAlg)
        sharedSecret = self._finishEphemeralDH(clientDHpublicKey)
        h = _kex.getHashProcessor(self.kexAlg)()
        h.update(NS(self.otherVersionString))
        h.update(NS(self.ourVersionString))
        h.update(NS(self.otherKexInitPayload))
        h.update(NS(self.ourKexInitPayload))
        h.update(NS(pubHostKey.blob()))
        h.update(self.dhGexRequest)
        h.update(MP(self.p))
        h.update(MP(self.g))
        h.update(MP(clientDHpublicKey))
        h.update(self.dhSecretKeyPublicMP)
        h.update(sharedSecret)
        exchangeHash = h.digest()
        self.sendPacket(MSG_KEX_DH_GEX_REPLY, NS(pubHostKey.blob()) + self.dhSecretKeyPublicMP + NS(privHostKey.sign(exchangeHash, signatureType=self.keyAlg)))
        self._keySetup(sharedSecret, exchangeHash)

    def _keySetup(self, sharedSecret, exchangeHash):
        """
        See SSHTransportBase._keySetup().
        """
        firstKey = self.sessionID is None
        SSHTransportBase._keySetup(self, sharedSecret, exchangeHash)
        if firstKey:
            self.sendExtInfo([(b'server-sig-algs', b','.join(self.supportedPublicKeys))])

    def ssh_NEWKEYS(self, packet):
        """
        Called when we get a MSG_NEWKEYS message.  No payload.
        When we get this, the keys have been set on both sides, and we
        start using them to encrypt and authenticate the connection.

        @type packet: L{bytes}
        @param packet: The message data.
        """
        if packet != b'':
            self.sendDisconnect(DISCONNECT_PROTOCOL_ERROR, b'NEWKEYS takes no data')
            return
        self._newKeys()

    def ssh_SERVICE_REQUEST(self, packet):
        """
        Called when we get a MSG_SERVICE_REQUEST message.  Payload::
            string serviceName

        The client has requested a service.  If we can start the service,
        start it; otherwise, disconnect with
        DISCONNECT_SERVICE_NOT_AVAILABLE.

        @type packet: L{bytes}
        @param packet: The message data.
        """
        service, rest = getNS(packet)
        cls = self.factory.getService(self, service)
        if not cls:
            self.sendDisconnect(DISCONNECT_SERVICE_NOT_AVAILABLE, b"don't have service " + service)
            return
        else:
            self.sendPacket(MSG_SERVICE_ACCEPT, NS(service))
            self.setService(cls())