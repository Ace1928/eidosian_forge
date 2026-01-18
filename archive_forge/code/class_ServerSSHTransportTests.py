import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class ServerSSHTransportTests(ServerSSHTransportBaseCase, TransportTestCase):
    """
    Tests for SSHServerTransport.
    """

    def test__getHostKeys(self):
        """
        L{transport.SSHServerTransport._getHostKeys} returns host keys from
        the factory, looked up by public key signature algorithm.
        """
        self.proto.factory.publicKeys = {b'ssh-rsa': keys.Key.fromString(keydata.publicRSA_openssh), b'ssh-dss': keys.Key.fromString(keydata.publicDSA_openssh), b'ecdsa-sha2-nistp256': keys.Key.fromString(keydata.publicECDSA_openssh), b'ssh-ed25519': keys.Key.fromString(keydata.publicEd25519_openssh)}
        self.proto.factory.privateKeys = {b'ssh-rsa': keys.Key.fromString(keydata.privateRSA_openssh), b'ssh-dss': keys.Key.fromString(keydata.privateDSA_openssh), b'ecdsa-sha2-nistp256': keys.Key.fromString(keydata.privateECDSA_openssh), b'ssh-ed25519': keys.Key.fromString(keydata.privateEd25519_openssh_new)}
        self.assertEqual(self.proto._getHostKeys(b'ssh-rsa'), (self.proto.factory.publicKeys[b'ssh-rsa'], self.proto.factory.privateKeys[b'ssh-rsa']))
        self.assertEqual(self.proto._getHostKeys(b'rsa-sha2-256'), (self.proto.factory.publicKeys[b'ssh-rsa'], self.proto.factory.privateKeys[b'ssh-rsa']))
        self.assertEqual(self.proto._getHostKeys(b'rsa-sha2-512'), (self.proto.factory.publicKeys[b'ssh-rsa'], self.proto.factory.privateKeys[b'ssh-rsa']))
        self.assertEqual(self.proto._getHostKeys(b'ssh-dss'), (self.proto.factory.publicKeys[b'ssh-dss'], self.proto.factory.privateKeys[b'ssh-dss']))
        self.assertEqual(self.proto._getHostKeys(b'ecdsa-sha2-nistp256'), (self.proto.factory.publicKeys[b'ecdsa-sha2-nistp256'], self.proto.factory.privateKeys[b'ecdsa-sha2-nistp256']))
        self.assertEqual(self.proto._getHostKeys(b'ssh-ed25519'), (self.proto.factory.publicKeys[b'ssh-ed25519'], self.proto.factory.privateKeys[b'ssh-ed25519']))
        self.assertRaises(KeyError, self.proto._getHostKeys, b'ecdsa-sha2-nistp384')

    def test_KEXINITMultipleAlgorithms(self):
        """
        Receiving a KEXINIT packet listing multiple supported algorithms will
        set up the first common algorithm found in the client's preference
        list.
        """
        self.proto.dataReceived(b'SSH-2.0-Twisted\r\n\x00\x00\x01\xf4\x04\x14\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x00\x00\x00bdiffie-hellman-group1-sha1,diffie-hellman-group-exchange-sha1,diffie-hellman-group-exchange-sha256\x00\x00\x00\x0fssh-dss,ssh-rsa\x00\x00\x00\x85aes128-ctr,aes128-cbc,aes192-ctr,aes192-cbc,aes256-ctr,aes256-cbc,cast128-ctr,cast128-cbc,blowfish-ctr,blowfish-cbc,3des-ctr,3des-cbc\x00\x00\x00\x85aes128-ctr,aes128-cbc,aes192-ctr,aes192-cbc,aes256-ctr,aes256-cbc,cast128-ctr,cast128-cbc,blowfish-ctr,blowfish-cbc,3des-ctr,3des-cbc\x00\x00\x00\x12hmac-md5,hmac-sha1\x00\x00\x00\x12hmac-md5,hmac-sha1\x00\x00\x00\tnone,zlib\x00\x00\x00\tnone,zlib\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x99\x99\x99\x99')
        self.assertEqual(self.proto.kexAlg, b'diffie-hellman-group-exchange-sha1')
        self.assertEqual(self.proto.keyAlg, b'ssh-dss')
        self.assertEqual(self.proto.outgoingCompressionType, b'none')
        self.assertEqual(self.proto.incomingCompressionType, b'none')
        self.assertFalse(self.proto._peerSupportsExtensions)
        ne = self.proto.nextEncryptions
        self.assertEqual(ne.outCipType, b'aes128-ctr')
        self.assertEqual(ne.inCipType, b'aes128-ctr')
        self.assertEqual(ne.outMACType, b'hmac-md5')
        self.assertEqual(ne.inMACType, b'hmac-md5')

    def test_KEXINITExtensionNegotiation(self):
        """
        If the client sends "ext-info-c" in its key exchange algorithms,
        then the server notes that the client supports extension
        negotiation.  See RFC 8308, section 2.1.
        """
        kexInitPacket = b'\x00' * 16 + common.NS(b'diffie-hellman-group-exchange-sha256,ext-info-c') + common.NS(b'ssh-rsa') + common.NS(b'aes256-ctr') + common.NS(b'aes256-ctr') + common.NS(b'hmac-sha1') + common.NS(b'hmac-sha1') + common.NS(b'none') + common.NS(b'none') + common.NS(b'') + common.NS(b'') + b'\x00' + b'\x00\x00\x00\x00'
        self.proto.ssh_KEXINIT(kexInitPacket)
        self.assertEqual(self.proto.kexAlg, b'diffie-hellman-group-exchange-sha256')
        self.assertTrue(self.proto._peerSupportsExtensions)

    def test_ignoreGuessPacketKex(self):
        """
        The client is allowed to send a guessed key exchange packet
        after it sends the KEXINIT packet.  However, if the key exchanges
        do not match, that guess packet must be ignored.  This tests that
        the packet is ignored in the case of the key exchange method not
        matching.
        """
        kexInitPacket = b'\x00' * 16 + b''.join([common.NS(x) for x in [b','.join(y) for y in [self.proto.supportedKeyExchanges[::-1], self.proto.supportedPublicKeys, self.proto.supportedCiphers, self.proto.supportedCiphers, self.proto.supportedMACs, self.proto.supportedMACs, self.proto.supportedCompressions, self.proto.supportedCompressions, self.proto.supportedLanguages, self.proto.supportedLanguages]]]) + b'\xff\x00\x00\x00\x00'
        self.proto.ssh_KEXINIT(kexInitPacket)
        self.assertTrue(self.proto.ignoreNextPacket)
        self.proto.ssh_DEBUG(b'\x01\x00\x00\x00\x04test\x00\x00\x00\x00')
        self.assertTrue(self.proto.ignoreNextPacket)
        self.proto.ssh_KEX_DH_GEX_REQUEST_OLD(b'\x00\x00\x08\x00')
        self.assertFalse(self.proto.ignoreNextPacket)
        self.assertEqual(self.packets, [])
        self.proto.ignoreNextPacket = True
        self.proto.ssh_KEX_DH_GEX_REQUEST(b'\x00\x00\x08\x00' * 3)
        self.assertFalse(self.proto.ignoreNextPacket)
        self.assertEqual(self.packets, [])

    def test_ignoreGuessPacketKey(self):
        """
        Like test_ignoreGuessPacketKex, but for an incorrectly guessed
        public key format.
        """
        kexInitPacket = b'\x00' * 16 + b''.join([common.NS(x) for x in [b','.join(y) for y in [self.proto.supportedKeyExchanges, self.proto.supportedPublicKeys[::-1], self.proto.supportedCiphers, self.proto.supportedCiphers, self.proto.supportedMACs, self.proto.supportedMACs, self.proto.supportedCompressions, self.proto.supportedCompressions, self.proto.supportedLanguages, self.proto.supportedLanguages]]]) + b'\xff\x00\x00\x00\x00'
        self.proto.ssh_KEXINIT(kexInitPacket)
        self.assertTrue(self.proto.ignoreNextPacket)
        self.proto.ssh_DEBUG(b'\x01\x00\x00\x00\x04test\x00\x00\x00\x00')
        self.assertTrue(self.proto.ignoreNextPacket)
        self.proto.ssh_KEX_DH_GEX_REQUEST_OLD(b'\x00\x00\x08\x00')
        self.assertFalse(self.proto.ignoreNextPacket)
        self.assertEqual(self.packets, [])
        self.proto.ignoreNextPacket = True
        self.proto.ssh_KEX_DH_GEX_REQUEST(b'\x00\x00\x08\x00' * 3)
        self.assertFalse(self.proto.ignoreNextPacket)
        self.assertEqual(self.packets, [])

    def assertKexDHInitResponse(self, kexAlgorithm, keyAlgorithm, bits):
        """
        Test that the KEXDH_INIT packet causes the server to send a
        KEXDH_REPLY with the server's public key and a signature.

        @param kexAlgorithm: The key exchange algorithm to use.
        @type kexAlgorithm: L{bytes}

        @param keyAlgorithm: The public key signature algorithm to use.
        @type keyAlgorithm: L{bytes}

        @param bits: The bit length of the DH modulus.
        @type bits: L{int}
        """
        self.proto.supportedKeyExchanges = [kexAlgorithm]
        self.proto.supportedPublicKeys = [keyAlgorithm]
        self.proto.dataReceived(self.transport.value())
        pubHostKey, privHostKey = self.proto._getHostKeys(keyAlgorithm)
        g, p = _kex.getDHGeneratorAndPrime(kexAlgorithm)
        e = pow(g, 5000, p)
        self.proto.ssh_KEX_DH_GEX_REQUEST_OLD(common.MP(e))
        y = common.getMP(common.NS(b'\x99' * (bits // 8)))[0]
        f = _MPpow(self.proto.g, y, self.proto.p)
        self.assertEqual(self.proto.dhSecretKeyPublicMP, f)
        sharedSecret = _MPpow(e, y, self.proto.p)
        h = sha1()
        h.update(common.NS(self.proto.ourVersionString) * 2)
        h.update(common.NS(self.proto.ourKexInitPayload) * 2)
        h.update(common.NS(pubHostKey.blob()))
        h.update(common.MP(e))
        h.update(f)
        h.update(sharedSecret)
        exchangeHash = h.digest()
        signature = privHostKey.sign(exchangeHash, signatureType=keyAlgorithm)
        self.assertEqual(self.packets, [(transport.MSG_KEXDH_REPLY, common.NS(pubHostKey.blob()) + f + common.NS(signature)), (transport.MSG_NEWKEYS, b'')])

    def test_checkBad_KEX_ECDH_INIT_CurveName(self):
        """
        Test that if the server receives a KEX_DH_GEX_REQUEST_OLD message
        and the key exchange algorithm is not set, we raise a ConchError.
        """
        self.proto.kexAlg = b'bad-curve'
        self.proto.keyAlg = b'ssh-rsa'
        self.assertRaises(UnsupportedAlgorithm, self.proto._ssh_KEX_ECDH_INIT, common.NS(b'unused-key'))

    def test_checkBad_KEX_INIT_CurveName(self):
        """
        Test that if the server received a bad name for a curve
        we raise an UnsupportedAlgorithm error.
        """
        kexmsg = b'\xaa' * 16 + common.NS(b'ecdh-sha2-nistp256') + common.NS(b'ssh-rsa') + common.NS(b'aes256-ctr') + common.NS(b'aes256-ctr') + common.NS(b'hmac-sha1') + common.NS(b'hmac-sha1') + common.NS(b'none') + common.NS(b'none') + common.NS(b'') + common.NS(b'') + b'\x00' + b'\x00\x00\x00\x00'
        self.proto.ssh_KEXINIT(kexmsg)
        self.assertRaises(AttributeError)
        self.assertRaises(UnsupportedAlgorithm)

    def test_KEXDH_INIT_GROUP14(self):
        """
        KEXDH_INIT messages are processed when the
        diffie-hellman-group14-sha1 key exchange algorithm and the ssh-rsa
        public key signature algorithm are requested.
        """
        self.assertKexDHInitResponse(b'diffie-hellman-group14-sha1', b'ssh-rsa', 2048)

    def test_KEXDH_INIT_GROUP14_rsa_sha2_256(self):
        """
        KEXDH_INIT messages are processed when the
        diffie-hellman-group14-sha1 key exchange algorithm and the
        rsa-sha2-256 public key signature algorithm are requested.
        """
        self.assertKexDHInitResponse(b'diffie-hellman-group14-sha1', b'rsa-sha2-256', 2048)

    def test_KEXDH_INIT_GROUP14_rsa_sha2_512(self):
        """
        KEXDH_INIT messages are processed when the
        diffie-hellman-group14-sha1 key exchange algorithm and the
        rsa-sha2-256 public key signature algorithm are requested.
        """
        self.assertKexDHInitResponse(b'diffie-hellman-group14-sha1', b'rsa-sha2-512', 2048)

    def test_keySetup(self):
        """
        Test that _keySetup sets up the next encryption keys.
        """
        self.proto.kexAlg = b'diffie-hellman-group14-sha1'
        self.proto.nextEncryptions = MockCipher()
        self.simulateKeyExchange(b'AB', b'CD')
        self.assertEqual(self.proto.sessionID, b'CD')
        self.simulateKeyExchange(b'AB', b'EF')
        self.assertEqual(self.proto.sessionID, b'CD')
        self.assertEqual(self.packets[-1], (transport.MSG_NEWKEYS, b''))
        newKeys = [self.proto._getKey(c, b'AB', b'EF') for c in iterbytes(b'ABCDEF')]
        self.assertEqual(self.proto.nextEncryptions.keys, (newKeys[1], newKeys[3], newKeys[0], newKeys[2], newKeys[5], newKeys[4]))

    def test_keySetupWithExtInfo(self):
        """
        If the client advertised support for extension negotiation, then
        _keySetup sends SSH_MSG_EXT_INFO with the "server-sig-algs"
        extension as the next packet following the server's first
        SSH_MSG_NEWKEYS.  See RFC 8308, sections 2.4 and 3.1.
        """
        self.proto.supportedPublicKeys = [b'ssh-rsa', b'rsa-sha2-256', b'rsa-sha2-512']
        self.proto.kexAlg = b'diffie-hellman-group14-sha1'
        self.proto.nextEncryptions = MockCipher()
        self.proto._peerSupportsExtensions = True
        self.simulateKeyExchange(b'AB', b'CD')
        self.assertEqual(self.packets[-2], (transport.MSG_NEWKEYS, b''))
        self.assertEqual(self.packets[-1], (transport.MSG_EXT_INFO, b'\x00\x00\x00\x01' + common.NS(b'server-sig-algs') + common.NS(b'ssh-rsa,rsa-sha2-256,rsa-sha2-512')))
        self.simulateKeyExchange(b'AB', b'EF')
        self.assertEqual(self.packets[-1], (transport.MSG_NEWKEYS, b''))

    def test_ECDH_keySetup(self):
        """
        Test that _keySetup sets up the next encryption keys.
        """
        self.proto.kexAlg = b'ecdh-sha2-nistp256'
        self.proto.nextEncryptions = MockCipher()
        self.simulateKeyExchange(b'AB', b'CD')
        self.assertEqual(self.proto.sessionID, b'CD')
        self.simulateKeyExchange(b'AB', b'EF')
        self.assertEqual(self.proto.sessionID, b'CD')
        self.assertEqual(self.packets[-1], (transport.MSG_NEWKEYS, b''))
        newKeys = [self.proto._getKey(c, b'AB', b'EF') for c in iterbytes(b'ABCDEF')]
        self.assertEqual(self.proto.nextEncryptions.keys, (newKeys[1], newKeys[3], newKeys[0], newKeys[2], newKeys[5], newKeys[4]))

    def test_NEWKEYS(self):
        """
        Test that NEWKEYS transitions the keys in nextEncryptions to
        currentEncryptions.
        """
        self.test_KEXINITMultipleAlgorithms()
        self.proto.nextEncryptions = transport.SSHCiphers(b'none', b'none', b'none', b'none')
        self.proto.ssh_NEWKEYS(b'')
        self.assertIs(self.proto.currentEncryptions, self.proto.nextEncryptions)
        self.assertIsNone(self.proto.outgoingCompression)
        self.assertIsNone(self.proto.incomingCompression)
        self.proto.outgoingCompressionType = b'zlib'
        self.simulateKeyExchange(b'AB', b'CD')
        self.proto.ssh_NEWKEYS(b'')
        self.assertIsNotNone(self.proto.outgoingCompression)
        self.proto.incomingCompressionType = b'zlib'
        self.simulateKeyExchange(b'AB', b'EF')
        self.proto.ssh_NEWKEYS(b'')
        self.assertIsNotNone(self.proto.incomingCompression)

    def test_SERVICE_REQUEST(self):
        """
        Test that the SERVICE_REQUEST message requests and starts a
        service.
        """
        self.proto.ssh_SERVICE_REQUEST(common.NS(b'ssh-userauth'))
        self.assertEqual(self.packets, [(transport.MSG_SERVICE_ACCEPT, common.NS(b'ssh-userauth'))])
        self.assertEqual(self.proto.service.name, b'MockService')

    def test_disconnectNEWKEYSData(self):
        """
        Test that NEWKEYS disconnects if it receives data.
        """
        self.proto.ssh_NEWKEYS(b'bad packet')
        self.checkDisconnected()

    def test_disconnectSERVICE_REQUESTBadService(self):
        """
        Test that SERVICE_REQUESTS disconnects if an unknown service is
        requested.
        """
        self.proto.ssh_SERVICE_REQUEST(common.NS(b'no service'))
        self.checkDisconnected(transport.DISCONNECT_SERVICE_NOT_AVAILABLE)