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
class SSHTransportBase(protocol.Protocol):
    """
    Protocol supporting basic SSH functionality: sending/receiving packets
    and message dispatch.  To connect to or run a server, you must use
    SSHClientTransport or SSHServerTransport.

    @ivar protocolVersion: A string representing the version of the SSH
        protocol we support.  Currently defaults to '2.0'.

    @ivar version: A string representing the version of the server or client.
        Currently defaults to 'Twisted'.

    @ivar comment: An optional string giving more information about the
        server or client.

    @ivar supportedCiphers: A list of strings representing the encryption
        algorithms supported, in order from most-preferred to least.

    @ivar supportedMACs: A list of strings representing the message
        authentication codes (hashes) supported, in order from most-preferred
        to least.  Both this and supportedCiphers can include 'none' to use
        no encryption or authentication, but that must be done manually,

    @ivar supportedKeyExchanges: A list of strings representing the
        key exchanges supported, in order from most-preferred to least.

    @ivar supportedPublicKeys:  A list of strings representing the
        public key algorithms supported, in order from most-preferred to
        least.

    @ivar supportedCompressions: A list of strings representing compression
        types supported, from most-preferred to least.

    @ivar supportedLanguages: A list of strings representing languages
        supported, from most-preferred to least.

    @ivar supportedVersions: A container of strings representing supported ssh
        protocol version numbers.

    @ivar isClient: A boolean indicating whether this is a client or server.

    @ivar gotVersion: A boolean indicating whether we have received the
        version string from the other side.

    @ivar buf: Data we've received but hasn't been parsed into a packet.

    @ivar outgoingPacketSequence: the sequence number of the next packet we
        will send.

    @ivar incomingPacketSequence: the sequence number of the next packet we
        are expecting from the other side.

    @ivar outgoingCompression: an object supporting the .compress(str) and
        .flush() methods, or None if there is no outgoing compression.  Used to
        compress outgoing data.

    @ivar outgoingCompressionType: A string representing the outgoing
        compression type.

    @ivar incomingCompression: an object supporting the .decompress(str)
        method, or None if there is no incoming compression.  Used to
        decompress incoming data.

    @ivar incomingCompressionType: A string representing the incoming
        compression type.

    @ivar ourVersionString: the version string that we sent to the other side.
        Used in the key exchange.

    @ivar otherVersionString: the version string sent by the other side.  Used
        in the key exchange.

    @ivar ourKexInitPayload: the MSG_KEXINIT payload we sent.  Used in the key
        exchange.

    @ivar otherKexInitPayload: the MSG_KEXINIT payload we received.  Used in
        the key exchange

    @ivar sessionID: a string that is unique to this SSH session.  Created as
        part of the key exchange, sessionID is used to generate the various
        encryption and authentication keys.

    @ivar service: an SSHService instance, or None.  If it's set to an object,
        it's the currently running service.

    @ivar kexAlg: the agreed-upon key exchange algorithm.

    @ivar keyAlg: the agreed-upon public key type for the key exchange.

    @ivar currentEncryptions: an SSHCiphers instance.  It represents the
        current encryption and authentication options for the transport.

    @ivar nextEncryptions: an SSHCiphers instance.  Held here until the
        MSG_NEWKEYS messages are exchanged, when nextEncryptions is
        transitioned to currentEncryptions.

    @ivar first: the first bytes of the next packet.  In order to avoid
        decrypting data twice, the first bytes are decrypted and stored until
        the whole packet is available.

    @ivar _keyExchangeState: The current protocol state with respect to key
        exchange.  This is either C{_KEY_EXCHANGE_NONE} if no key exchange is
        in progress (and returns to this value after any key exchange
        completqes), C{_KEY_EXCHANGE_REQUESTED} if this side of the connection
        initiated a key exchange, and C{_KEY_EXCHANGE_PROGRESSING} if the other
        side of the connection initiated a key exchange.  C{_KEY_EXCHANGE_NONE}
        is the initial value (however SSH connections begin with key exchange,
        so it will quickly change to another state).

    @ivar _blockedByKeyExchange: Whenever C{_keyExchangeState} is not
        C{_KEY_EXCHANGE_NONE}, this is a C{list} of pending messages which were
        passed to L{sendPacket} but could not be sent because it is not legal
        to send them while a key exchange is in progress.  When the key
        exchange completes, another attempt is made to send these messages.

    @ivar _peerSupportsExtensions: a boolean indicating whether the other side
        of the connection supports RFC 8308 extension negotiation.

    @ivar peerExtensions: a dict of extensions supported by the other side of
        the connection.
    """
    _log = Logger()
    protocolVersion = b'2.0'
    version = b'Twisted_' + twisted_version.encode('ascii')
    comment = b''
    ourVersionString = (b'SSH-' + protocolVersion + b'-' + version + b' ' + comment).strip()
    supportedCiphers = _getSupportedCiphers()
    supportedMACs = [b'hmac-sha2-512', b'hmac-sha2-384', b'hmac-sha2-256', b'hmac-sha1', b'hmac-md5']
    supportedKeyExchanges = _kex.getSupportedKeyExchanges()
    supportedPublicKeys = []
    for eckey in supportedKeyExchanges:
        if eckey.find(b'ecdh') != -1:
            supportedPublicKeys += [eckey.replace(b'ecdh', b'ecdsa')]
    supportedPublicKeys += [b'rsa-sha2-512', b'rsa-sha2-256', b'ssh-rsa', b'ssh-dss']
    if default_backend().ed25519_supported():
        supportedPublicKeys.append(b'ssh-ed25519')
    supportedCompressions = [b'none', b'zlib']
    supportedLanguages = ()
    supportedVersions = (b'1.99', b'2.0')
    isClient = False
    gotVersion = False
    buf = b''
    outgoingPacketSequence = 0
    incomingPacketSequence = 0
    outgoingCompression = None
    incomingCompression = None
    sessionID = None
    service = None
    _KEY_EXCHANGE_NONE = '_KEY_EXCHANGE_NONE'
    _KEY_EXCHANGE_REQUESTED = '_KEY_EXCHANGE_REQUESTED'
    _KEY_EXCHANGE_PROGRESSING = '_KEY_EXCHANGE_PROGRESSING'
    _keyExchangeState = _KEY_EXCHANGE_NONE
    _blockedByKeyExchange = None
    _EXT_INFO_C = b'ext-info-c'
    _EXT_INFO_S = b'ext-info-s'
    _peerSupportsExtensions = False
    peerExtensions: Dict[bytes, bytes] = {}

    def connectionLost(self, reason):
        """
        When the underlying connection is closed, stop the running service (if
        any), and log out the avatar (if any).

        @type reason: L{twisted.python.failure.Failure}
        @param reason: The cause of the connection being closed.
        """
        if self.service:
            self.service.serviceStopped()
        if hasattr(self, 'avatar'):
            self.logoutFunction()
        self._log.info('connection lost')

    def connectionMade(self):
        """
        Called when the connection is made to the other side.  We sent our
        version and the MSG_KEXINIT packet.
        """
        self.transport.write(self.ourVersionString + b'\r\n')
        self.currentEncryptions = SSHCiphers(b'none', b'none', b'none', b'none')
        self.currentEncryptions.setKeys(b'', b'', b'', b'', b'', b'')
        self.sendKexInit()

    def sendKexInit(self):
        """
        Send a I{KEXINIT} message to initiate key exchange or to respond to a
        key exchange initiated by the peer.

        @raise RuntimeError: If a key exchange has already been started and it
            is not appropriate to send a I{KEXINIT} message at this time.

        @return: L{None}
        """
        if self._keyExchangeState != self._KEY_EXCHANGE_NONE:
            raise RuntimeError('Cannot send KEXINIT while key exchange state is %r' % (self._keyExchangeState,))
        supportedKeyExchanges = list(self.supportedKeyExchanges)
        supportedKeyExchanges.append(self._EXT_INFO_C if self.isClient else self._EXT_INFO_S)
        self.ourKexInitPayload = b''.join([bytes((MSG_KEXINIT,)), randbytes.secureRandom(16), NS(b','.join(supportedKeyExchanges)), NS(b','.join(self.supportedPublicKeys)), NS(b','.join(self.supportedCiphers)), NS(b','.join(self.supportedCiphers)), NS(b','.join(self.supportedMACs)), NS(b','.join(self.supportedMACs)), NS(b','.join(self.supportedCompressions)), NS(b','.join(self.supportedCompressions)), NS(b','.join(self.supportedLanguages)), NS(b','.join(self.supportedLanguages)), b'\x00\x00\x00\x00\x00'])
        self.sendPacket(MSG_KEXINIT, self.ourKexInitPayload[1:])
        self._keyExchangeState = self._KEY_EXCHANGE_REQUESTED
        self._blockedByKeyExchange = []

    def _allowedKeyExchangeMessageType(self, messageType):
        """
        Determine if the given message type may be sent while key exchange is
        in progress.

        @param messageType: The type of message
        @type messageType: L{int}

        @return: C{True} if the given type of message may be sent while key
            exchange is in progress, C{False} if it may not.
        @rtype: L{bool}

        @see: U{http://tools.ietf.org/html/rfc4253#section-7.1}
        """
        if 1 <= messageType <= 19:
            return messageType not in (MSG_SERVICE_REQUEST, MSG_SERVICE_ACCEPT, MSG_EXT_INFO)
        if 20 <= messageType <= 29:
            return messageType not in (MSG_KEXINIT,)
        return 30 <= messageType <= 49

    def sendPacket(self, messageType, payload):
        """
        Sends a packet.  If it's been set up, compress the data, encrypt it,
        and authenticate it before sending.  If key exchange is in progress and
        the message is not part of key exchange, queue it to be sent later.

        @param messageType: The type of the packet; generally one of the
                            MSG_* values.
        @type messageType: L{int}
        @param payload: The payload for the message.
        @type payload: L{str}
        """
        if self._keyExchangeState != self._KEY_EXCHANGE_NONE:
            if not self._allowedKeyExchangeMessageType(messageType):
                self._blockedByKeyExchange.append((messageType, payload))
                return
        payload = bytes((messageType,)) + payload
        if self.outgoingCompression:
            payload = self.outgoingCompression.compress(payload) + self.outgoingCompression.flush(2)
        bs = self.currentEncryptions.encBlockSize
        totalSize = 5 + len(payload)
        lenPad = bs - totalSize % bs
        if lenPad < 4:
            lenPad = lenPad + bs
        packet = struct.pack('!LB', totalSize + lenPad - 4, lenPad) + payload + randbytes.secureRandom(lenPad)
        encPacket = self.currentEncryptions.encrypt(packet) + self.currentEncryptions.makeMAC(self.outgoingPacketSequence, packet)
        self.transport.write(encPacket)
        self.outgoingPacketSequence += 1

    def getPacket(self):
        """
        Try to return a decrypted, authenticated, and decompressed packet
        out of the buffer.  If there is not enough data, return None.

        @rtype: L{str} or L{None}
        @return: The decoded packet, if any.
        """
        bs = self.currentEncryptions.decBlockSize
        ms = self.currentEncryptions.verifyDigestSize
        if len(self.buf) < bs:
            return
        if not hasattr(self, 'first'):
            first = self.currentEncryptions.decrypt(self.buf[:bs])
        else:
            first = self.first
            del self.first
        packetLen, paddingLen = struct.unpack('!LB', first[:5])
        if packetLen > 1048576:
            self.sendDisconnect(DISCONNECT_PROTOCOL_ERROR, networkString(f'bad packet length {packetLen}'))
            return
        if len(self.buf) < packetLen + 4 + ms:
            self.first = first
            return
        if (packetLen + 4) % bs != 0:
            self.sendDisconnect(DISCONNECT_PROTOCOL_ERROR, networkString('bad packet mod (%i%%%i == %i)' % (packetLen + 4, bs, (packetLen + 4) % bs)))
            return
        encData, self.buf = (self.buf[:4 + packetLen], self.buf[4 + packetLen:])
        packet = first + self.currentEncryptions.decrypt(encData[bs:])
        if len(packet) != 4 + packetLen:
            self.sendDisconnect(DISCONNECT_PROTOCOL_ERROR, b'bad decryption')
            return
        if ms:
            macData, self.buf = (self.buf[:ms], self.buf[ms:])
            if not self.currentEncryptions.verify(self.incomingPacketSequence, packet, macData):
                self.sendDisconnect(DISCONNECT_MAC_ERROR, b'bad MAC')
                return
        payload = packet[5:-paddingLen]
        if self.incomingCompression:
            try:
                payload = self.incomingCompression.decompress(payload)
            except Exception:
                self._log.failure('Error decompressing payload')
                self.sendDisconnect(DISCONNECT_COMPRESSION_ERROR, b'compression error')
                return
        self.incomingPacketSequence += 1
        return payload

    def _unsupportedVersionReceived(self, remoteVersion):
        """
        Called when an unsupported version of the ssh protocol is received from
        the remote endpoint.

        @param remoteVersion: remote ssh protocol version which is unsupported
            by us.
        @type remoteVersion: L{str}
        """
        self.sendDisconnect(DISCONNECT_PROTOCOL_VERSION_NOT_SUPPORTED, b'bad version ' + remoteVersion)

    def dataReceived(self, data):
        """
        First, check for the version string (SSH-2.0-*).  After that has been
        received, this method adds data to the buffer, and pulls out any
        packets.

        @type data: L{bytes}
        @param data: The data that was received.
        """
        self.buf = self.buf + data
        if not self.gotVersion:
            if len(self.buf) > 4096:
                self.sendDisconnect(DISCONNECT_CONNECTION_LOST, b'Peer version string longer than 4KB. Preventing a denial of service attack.')
                return
            if self.buf.find(b'\n', self.buf.find(b'SSH-')) == -1:
                return
            lines = self.buf.split(b'\n')
            for p in lines:
                if p.startswith(b'SSH-'):
                    self.gotVersion = True
                    self.otherVersionString = p.rstrip(b'\r')
                    remoteVersion = p.split(b'-')[1]
                    if remoteVersion not in self.supportedVersions:
                        self._unsupportedVersionReceived(remoteVersion)
                        return
                    i = lines.index(p)
                    self.buf = b'\n'.join(lines[i + 1:])
        packet = self.getPacket()
        while packet:
            messageNum = ord(packet[0:1])
            self.dispatchMessage(messageNum, packet[1:])
            packet = self.getPacket()

    def dispatchMessage(self, messageNum, payload):
        """
        Send a received message to the appropriate method.

        @type messageNum: L{int}
        @param messageNum: The message number.

        @type payload: L{bytes}
        @param payload: The message payload.
        """
        if messageNum < 50 and messageNum in messages:
            messageType = messages[messageNum][4:]
            f = getattr(self, f'ssh_{messageType}', None)
            if f is not None:
                f(payload)
            else:
                self._log.debug("couldn't handle {messageType}: {payload!r}", messageType=messageType, payload=payload)
                self.sendUnimplemented()
        elif self.service:
            self.service.packetReceived(messageNum, payload)
        else:
            self._log.debug("couldn't handle {messageNum}: {payload!r}", messageNum=messageNum, payload=payload)
            self.sendUnimplemented()

    def getPeer(self):
        """
        Returns an L{SSHTransportAddress} corresponding to the other (peer)
        side of this transport.

        @return: L{SSHTransportAddress} for the peer
        @rtype: L{SSHTransportAddress}
        @since: 12.1
        """
        return address.SSHTransportAddress(self.transport.getPeer())

    def getHost(self):
        """
        Returns an L{SSHTransportAddress} corresponding to the this side of
        transport.

        @return: L{SSHTransportAddress} for the peer
        @rtype: L{SSHTransportAddress}
        @since: 12.1
        """
        return address.SSHTransportAddress(self.transport.getHost())

    @property
    def kexAlg(self):
        """
        The key exchange algorithm name agreed between client and server.
        """
        return self._kexAlg

    @kexAlg.setter
    def kexAlg(self, value):
        """
        Set the key exchange algorithm name.
        """
        self._kexAlg = value

    def ssh_KEXINIT(self, packet):
        """
        Called when we receive a MSG_KEXINIT message.  Payload::
            bytes[16] cookie
            string keyExchangeAlgorithms
            string keyAlgorithms
            string incomingEncryptions
            string outgoingEncryptions
            string incomingAuthentications
            string outgoingAuthentications
            string incomingCompressions
            string outgoingCompressions
            string incomingLanguages
            string outgoingLanguages
            bool firstPacketFollows
            unit32 0 (reserved)

        Starts setting up the key exchange, keys, encryptions, and
        authentications.  Extended by ssh_KEXINIT in SSHServerTransport and
        SSHClientTransport.

        @type packet: L{bytes}
        @param packet: The message data.

        @return: A L{tuple} of negotiated key exchange algorithms, key
        algorithms, and unhandled data, or L{None} if something went wrong.
        """
        self.otherKexInitPayload = bytes((MSG_KEXINIT,)) + packet
        k = getNS(packet[16:], 10)
        strings, rest = (k[:-1], k[-1])
        kexAlgs, keyAlgs, encCS, encSC, macCS, macSC, compCS, compSC, langCS, langSC = (s.split(b',') for s in strings)
        outs = [encSC, macSC, compSC]
        ins = [encCS, macCS, compCS]
        if self.isClient:
            outs, ins = (ins, outs)
        server = (self.supportedKeyExchanges, self.supportedPublicKeys, self.supportedCiphers, self.supportedCiphers, self.supportedMACs, self.supportedMACs, self.supportedCompressions, self.supportedCompressions)
        client = (kexAlgs, keyAlgs, outs[0], ins[0], outs[1], ins[1], outs[2], ins[2])
        if self.isClient:
            server, client = (client, server)
        self.kexAlg = ffs(client[0], server[0])
        self.keyAlg = ffs(client[1], server[1])
        self.nextEncryptions = SSHCiphers(ffs(client[2], server[2]), ffs(client[3], server[3]), ffs(client[4], server[4]), ffs(client[5], server[5]))
        self.outgoingCompressionType = ffs(client[6], server[6])
        self.incomingCompressionType = ffs(client[7], server[7])
        if None in (self.kexAlg, self.keyAlg, self.outgoingCompressionType, self.incomingCompressionType) or self.kexAlg in (self._EXT_INFO_C, self._EXT_INFO_S):
            self.sendDisconnect(DISCONNECT_KEY_EXCHANGE_FAILED, b"couldn't match all kex parts")
            return
        if None in self.nextEncryptions.__dict__.values():
            self.sendDisconnect(DISCONNECT_KEY_EXCHANGE_FAILED, b"couldn't match all kex parts")
            return
        self._peerSupportsExtensions = (self._EXT_INFO_S if self.isClient else self._EXT_INFO_C) in kexAlgs
        self._log.debug('kex alg={kexAlg!r} key alg={keyAlg!r}', kexAlg=self.kexAlg, keyAlg=self.keyAlg)
        self._log.debug('outgoing: {cip!r} {mac!r} {compression!r}', cip=self.nextEncryptions.outCipType, mac=self.nextEncryptions.outMACType, compression=self.outgoingCompressionType)
        self._log.debug('incoming: {cip!r} {mac!r} {compression!r}', cip=self.nextEncryptions.inCipType, mac=self.nextEncryptions.inMACType, compression=self.incomingCompressionType)
        if self._keyExchangeState == self._KEY_EXCHANGE_REQUESTED:
            self._keyExchangeState = self._KEY_EXCHANGE_PROGRESSING
        else:
            self.sendKexInit()
        return (kexAlgs, keyAlgs, rest)

    def ssh_DISCONNECT(self, packet):
        """
        Called when we receive a MSG_DISCONNECT message.  Payload::
            long code
            string description

        This means that the other side has disconnected.  Pass the message up
        and disconnect ourselves.

        @type packet: L{bytes}
        @param packet: The message data.
        """
        reasonCode = struct.unpack('>L', packet[:4])[0]
        description, foo = getNS(packet[4:])
        self.receiveError(reasonCode, description)
        self.transport.loseConnection()

    def ssh_IGNORE(self, packet):
        """
        Called when we receive a MSG_IGNORE message.  No payload.
        This means nothing; we simply return.

        @type packet: L{bytes}
        @param packet: The message data.
        """

    def ssh_UNIMPLEMENTED(self, packet):
        """
        Called when we receive a MSG_UNIMPLEMENTED message.  Payload::
            long packet

        This means that the other side did not implement one of our packets.

        @type packet: L{bytes}
        @param packet: The message data.
        """
        seqnum, = struct.unpack('>L', packet)
        self.receiveUnimplemented(seqnum)

    def ssh_DEBUG(self, packet):
        """
        Called when we receive a MSG_DEBUG message.  Payload::
            bool alwaysDisplay
            string message
            string language

        This means the other side has passed along some debugging info.

        @type packet: L{bytes}
        @param packet: The message data.
        """
        alwaysDisplay = bool(ord(packet[0:1]))
        message, lang, foo = getNS(packet[1:], 2)
        self.receiveDebug(alwaysDisplay, message, lang)

    def ssh_EXT_INFO(self, packet):
        """
        Called when we get a MSG_EXT_INFO message.  Payload::
            uint32 nr-extensions
            repeat the following 2 fields "nr-extensions" times:
              string extension-name
              string extension-value (binary)

        @type packet: L{bytes}
        @param packet: The message data.
        """
        numExtensions, = struct.unpack('>L', packet[:4])
        packet = packet[4:]
        extensions = {}
        for _ in range(numExtensions):
            extName, extValue, packet = getNS(packet, 2)
            extensions[extName] = extValue
        self.peerExtensions = extensions

    def setService(self, service):
        """
        Set our service to service and start it running.  If we were
        running a service previously, stop it first.

        @type service: C{SSHService}
        @param service: The service to attach.
        """
        self._log.debug('starting service {service!r}', service=service.name)
        if self.service:
            self.service.serviceStopped()
        self.service = service
        service.transport = self
        self.service.serviceStarted()

    def sendDebug(self, message, alwaysDisplay=False, language=b''):
        """
        Send a debug message to the other side.

        @param message: the message to send.
        @type message: L{str}
        @param alwaysDisplay: if True, tell the other side to always
                              display this message.
        @type alwaysDisplay: L{bool}
        @param language: optionally, the language the message is in.
        @type language: L{str}
        """
        self.sendPacket(MSG_DEBUG, (b'\x01' if alwaysDisplay else b'\x00') + NS(message) + NS(language))

    def sendIgnore(self, message):
        """
        Send a message that will be ignored by the other side.  This is
        useful to fool attacks based on guessing packet sizes in the
        encrypted stream.

        @param message: data to send with the message
        @type message: L{str}
        """
        self.sendPacket(MSG_IGNORE, NS(message))

    def sendUnimplemented(self):
        """
        Send a message to the other side that the last packet was not
        understood.
        """
        seqnum = self.incomingPacketSequence
        self.sendPacket(MSG_UNIMPLEMENTED, struct.pack('!L', seqnum))

    def sendDisconnect(self, reason, desc):
        """
        Send a disconnect message to the other side and then disconnect.

        @param reason: the reason for the disconnect.  Should be one of the
                       DISCONNECT_* values.
        @type reason: L{int}
        @param desc: a descrption of the reason for the disconnection.
        @type desc: L{str}
        """
        self.sendPacket(MSG_DISCONNECT, struct.pack('>L', reason) + NS(desc) + NS(b''))
        self._log.info('Disconnecting with error, code {code}\nreason: {description}', code=reason, description=desc)
        self.transport.loseConnection()

    def sendExtInfo(self, extensions):
        """
        Send an RFC 8308 extension advertisement to the remote peer.

        Nothing is sent if the peer doesn't support negotiations.
        @type extensions: L{list} of (L{bytes}, L{bytes})
        @param extensions: a list of (extension-name, extension-value) pairs.
        """
        if self._peerSupportsExtensions:
            payload = b''.join([struct.pack('>L', len(extensions))] + [NS(name) + NS(value) for name, value in extensions])
            self.sendPacket(MSG_EXT_INFO, payload)

    def _startEphemeralDH(self):
        """
        Prepares for a Diffie-Hellman key agreement exchange.

        Creates an ephemeral keypair in the group defined by (self.g,
        self.p) and stores it.
        """
        numbers = dh.DHParameterNumbers(self.p, self.g)
        parameters = numbers.parameters(default_backend())
        self.dhSecretKey = parameters.generate_private_key()
        y = self.dhSecretKey.public_key().public_numbers().y
        self.dhSecretKeyPublicMP = MP(y)

    def _finishEphemeralDH(self, remoteDHpublicKey):
        """
        Completes the Diffie-Hellman key agreement started by
        _startEphemeralDH, and forgets the ephemeral secret key.

        @type remoteDHpublicKey: L{int}
        @rtype: L{bytes}
        @return: The new shared secret, in SSH C{mpint} format.

        """
        remoteKey = dh.DHPublicNumbers(remoteDHpublicKey, dh.DHParameterNumbers(self.p, self.g)).public_key(default_backend())
        secret = self.dhSecretKey.exchange(remoteKey)
        del self.dhSecretKey
        secret = secret.lstrip(b'\x00')
        ch = ord(secret[0:1])
        if ch & 128:
            prefix = struct.pack('>L', len(secret) + 1) + b'\x00'
        else:
            prefix = struct.pack('>L', len(secret))
        return prefix + secret

    def _getKey(self, c, sharedSecret, exchangeHash):
        """
        Get one of the keys for authentication/encryption.

        @type c: L{bytes}
        @param c: The letter identifying which key this is.

        @type sharedSecret: L{bytes}
        @param sharedSecret: The shared secret K.

        @type exchangeHash: L{bytes}
        @param exchangeHash: The hash H from key exchange.

        @rtype: L{bytes}
        @return: The derived key.
        """
        hashProcessor = _kex.getHashProcessor(self.kexAlg)
        k1 = hashProcessor(sharedSecret + exchangeHash + c + self.sessionID)
        k1 = k1.digest()
        k2 = hashProcessor(sharedSecret + exchangeHash + k1).digest()
        k3 = hashProcessor(sharedSecret + exchangeHash + k1 + k2).digest()
        k4 = hashProcessor(sharedSecret + exchangeHash + k1 + k2 + k3).digest()
        return k1 + k2 + k3 + k4

    def _keySetup(self, sharedSecret, exchangeHash):
        """
        Set up the keys for the connection and sends MSG_NEWKEYS when
        finished,

        @param sharedSecret: a secret string agreed upon using a Diffie-
                             Hellman exchange, so it is only shared between
                             the server and the client.
        @type sharedSecret: L{str}
        @param exchangeHash: A hash of various data known by both sides.
        @type exchangeHash: L{str}
        """
        if not self.sessionID:
            self.sessionID = exchangeHash
        initIVCS = self._getKey(b'A', sharedSecret, exchangeHash)
        initIVSC = self._getKey(b'B', sharedSecret, exchangeHash)
        encKeyCS = self._getKey(b'C', sharedSecret, exchangeHash)
        encKeySC = self._getKey(b'D', sharedSecret, exchangeHash)
        integKeyCS = self._getKey(b'E', sharedSecret, exchangeHash)
        integKeySC = self._getKey(b'F', sharedSecret, exchangeHash)
        outs = [initIVSC, encKeySC, integKeySC]
        ins = [initIVCS, encKeyCS, integKeyCS]
        if self.isClient:
            outs, ins = (ins, outs)
        self.nextEncryptions.setKeys(outs[0], outs[1], ins[0], ins[1], outs[2], ins[2])
        self.sendPacket(MSG_NEWKEYS, b'')

    def _newKeys(self):
        """
        Called back by a subclass once a I{MSG_NEWKEYS} message has been
        received.  This indicates key exchange has completed and new encryption
        and compression parameters should be adopted.  Any messages which were
        queued during key exchange will also be flushed.
        """
        self._log.debug('NEW KEYS')
        self.currentEncryptions = self.nextEncryptions
        if self.outgoingCompressionType == b'zlib':
            self.outgoingCompression = zlib.compressobj(6)
        if self.incomingCompressionType == b'zlib':
            self.incomingCompression = zlib.decompressobj()
        self._keyExchangeState = self._KEY_EXCHANGE_NONE
        messages = self._blockedByKeyExchange
        self._blockedByKeyExchange = None
        for messageType, payload in messages:
            self.sendPacket(messageType, payload)

    def isEncrypted(self, direction='out'):
        """
        Check if the connection is encrypted in the given direction.

        @type direction: L{str}
        @param direction: The direction: one of 'out', 'in', or 'both'.

        @rtype: L{bool}
        @return: C{True} if it is encrypted.
        """
        if direction == 'out':
            return self.currentEncryptions.outCipType != b'none'
        elif direction == 'in':
            return self.currentEncryptions.inCipType != b'none'
        elif direction == 'both':
            return self.isEncrypted('in') and self.isEncrypted('out')
        else:
            raise TypeError('direction must be "out", "in", or "both"')

    def isVerified(self, direction='out'):
        """
        Check if the connection is verified/authentication in the given direction.

        @type direction: L{str}
        @param direction: The direction: one of 'out', 'in', or 'both'.

        @rtype: L{bool}
        @return: C{True} if it is verified.
        """
        if direction == 'out':
            return self.currentEncryptions.outMACType != b'none'
        elif direction == 'in':
            return self.currentEncryptions.inMACType != b'none'
        elif direction == 'both':
            return self.isVerified('in') and self.isVerified('out')
        else:
            raise TypeError('direction must be "out", "in", or "both"')

    def loseConnection(self):
        """
        Lose the connection to the other side, sending a
        DISCONNECT_CONNECTION_LOST message.
        """
        self.sendDisconnect(DISCONNECT_CONNECTION_LOST, b'user closed connection')

    def receiveError(self, reasonCode, description):
        """
        Called when we receive a disconnect error message from the other
        side.

        @param reasonCode: the reason for the disconnect, one of the
                           DISCONNECT_ values.
        @type reasonCode: L{int}
        @param description: a human-readable description of the
                            disconnection.
        @type description: L{str}
        """
        self._log.error('Got remote error, code {code}\nreason: {description}', code=reasonCode, description=description)

    def receiveUnimplemented(self, seqnum):
        """
        Called when we receive an unimplemented packet message from the other
        side.

        @param seqnum: the sequence number that was not understood.
        @type seqnum: L{int}
        """
        self._log.warn('other side unimplemented packet #{seqnum}', seqnum=seqnum)

    def receiveDebug(self, alwaysDisplay, message, lang):
        """
        Called when we receive a debug message from the other side.

        @param alwaysDisplay: if True, this message should always be
                              displayed.
        @type alwaysDisplay: L{bool}
        @param message: the debug message
        @type message: L{str}
        @param lang: optionally the language the message is in.
        @type lang: L{str}
        """
        if alwaysDisplay:
            self._log.debug('Remote Debug Message: {message}', message=message)

    def _generateECPrivateKey(self):
        """
        Generate an private key for ECDH key exchange.

        @rtype: The appropriate private key type matching C{self.kexAlg}:
            L{ec.EllipticCurvePrivateKey} for C{ecdh-sha2-nistp*}, or
            L{x25519.X25519PrivateKey} for C{curve25519-sha256}.
        @return: The generated private key.
        """
        if self.kexAlg.startswith(b'ecdh-sha2-nistp'):
            try:
                curve = keys._curveTable[b'ecdsa' + self.kexAlg[4:]]
            except KeyError:
                raise UnsupportedAlgorithm('unused-key')
            return ec.generate_private_key(curve, default_backend())
        elif self.kexAlg in (b'curve25519-sha256', b'curve25519-sha256@libssh.org'):
            return x25519.X25519PrivateKey.generate()
        else:
            raise UnsupportedAlgorithm('Cannot generate elliptic curve private key for {!r}'.format(self.kexAlg))

    def _encodeECPublicKey(self, ecPub):
        """
        Encode an elliptic curve public key to bytes.

        @type ecPub: The appropriate public key type matching
            C{self.kexAlg}: L{ec.EllipticCurvePublicKey} for
            C{ecdh-sha2-nistp*}, or L{x25519.X25519PublicKey} for
            C{curve25519-sha256}.
        @param ecPub: The public key to encode.

        @rtype: L{bytes}
        @return: The encoded public key.
        """
        if self.kexAlg.startswith(b'ecdh-sha2-nistp'):
            return ecPub.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        elif self.kexAlg in (b'curve25519-sha256', b'curve25519-sha256@libssh.org'):
            return ecPub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        else:
            raise UnsupportedAlgorithm(f'Cannot encode elliptic curve public key for {self.kexAlg!r}')

    def _generateECSharedSecret(self, ecPriv, theirECPubBytes):
        """
        Generate a shared secret for ECDH key exchange.

        @type ecPriv: The appropriate private key type matching
            C{self.kexAlg}: L{ec.EllipticCurvePrivateKey} for
            C{ecdh-sha2-nistp*}, or L{x25519.X25519PrivateKey} for
            C{curve25519-sha256}.
        @param ecPriv: Our private key.

        @rtype: L{bytes}
        @return: The generated shared secret, as an SSH multiple-precision
            integer.
        """
        if self.kexAlg.startswith(b'ecdh-sha2-nistp'):
            try:
                curve = keys._curveTable[b'ecdsa' + self.kexAlg[4:]]
            except KeyError:
                raise UnsupportedAlgorithm('unused-key')
            theirECPub = ec.EllipticCurvePublicKey.from_encoded_point(curve, theirECPubBytes)
            sharedSecret = ecPriv.exchange(ec.ECDH(), theirECPub)
        elif self.kexAlg in (b'curve25519-sha256', b'curve25519-sha256@libssh.org'):
            theirECPub = x25519.X25519PublicKey.from_public_bytes(theirECPubBytes)
            sharedSecret = ecPriv.exchange(theirECPub)
        else:
            raise UnsupportedAlgorithm('Cannot generate elliptic curve shared secret for {!r}'.format(self.kexAlg))
        return _mpFromBytes(sharedSecret)