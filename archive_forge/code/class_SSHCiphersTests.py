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
class SSHCiphersTests(TestCase):
    """
    Tests for the SSHCiphers helper class.
    """
    if dependencySkip:
        skip = dependencySkip

    def test_init(self):
        """
        Test that the initializer sets up the SSHCiphers object.
        """
        ciphers = transport.SSHCiphers(b'A', b'B', b'C', b'D')
        self.assertEqual(ciphers.outCipType, b'A')
        self.assertEqual(ciphers.inCipType, b'B')
        self.assertEqual(ciphers.outMACType, b'C')
        self.assertEqual(ciphers.inMACType, b'D')

    def test_getCipher(self):
        """
        Test that the _getCipher method returns the correct cipher.
        """
        ciphers = transport.SSHCiphers(b'A', b'B', b'C', b'D')
        iv = key = b'\x00' * 16
        for cipName, (algClass, keySize, counter) in ciphers.cipherMap.items():
            cip = ciphers._getCipher(cipName, iv, key)
            if cipName == b'none':
                self.assertIsInstance(cip, transport._DummyCipher)
            else:
                self.assertIsInstance(cip.algorithm, algClass)

    def test_setKeysCiphers(self):
        """
        Test that setKeys sets up the ciphers.
        """
        key = b'\x00' * 64
        for cipName in transport.SSHTransportBase.supportedCiphers:
            modName, keySize, counter = transport.SSHCiphers.cipherMap[cipName]
            encCipher = transport.SSHCiphers(cipName, b'none', b'none', b'none')
            decCipher = transport.SSHCiphers(b'none', cipName, b'none', b'none')
            cip = encCipher._getCipher(cipName, key, key)
            bs = cip.algorithm.block_size // 8
            encCipher.setKeys(key, key, b'', b'', b'', b'')
            decCipher.setKeys(b'', b'', key, key, b'', b'')
            self.assertEqual(encCipher.encBlockSize, bs)
            self.assertEqual(decCipher.decBlockSize, bs)
            encryptor = cip.encryptor()
            enc = encryptor.update(key[:bs])
            enc2 = encryptor.update(key[:bs])
            self.assertEqual(encCipher.encrypt(key[:bs]), enc)
            self.assertEqual(encCipher.encrypt(key[:bs]), enc2)
            self.assertEqual(decCipher.decrypt(enc), key[:bs])
            self.assertEqual(decCipher.decrypt(enc2), key[:bs])

    def test_setKeysMACs(self):
        """
        Test that setKeys sets up the MACs.
        """
        key = b'\x00' * 64
        for macName, mod in transport.SSHCiphers.macMap.items():
            outMac = transport.SSHCiphers(b'none', b'none', macName, b'none')
            inMac = transport.SSHCiphers(b'none', b'none', b'none', macName)
            outMac.setKeys(b'', b'', b'', b'', key, b'')
            inMac.setKeys(b'', b'', b'', b'', b'', key)
            if mod:
                ds = mod().digest_size
            else:
                ds = 0
            self.assertEqual(inMac.verifyDigestSize, ds)
            if mod:
                mod, i, o, ds = outMac._getMAC(macName, key)
            seqid = 0
            data = key
            packet = b'\x00' * 4 + key
            if mod:
                mac = mod(o + mod(i + packet).digest()).digest()
            else:
                mac = b''
            self.assertEqual(outMac.makeMAC(seqid, data), mac)
            self.assertTrue(inMac.verify(seqid, data, mac))

    def test_makeMAC(self):
        """
        L{SSHCiphers.makeMAC} computes the HMAC of an outgoing SSH message with
        a particular sequence id and content data.
        """
        vectors = [(b'\x0b' * 16, b'Hi There', b'9294727a3638bb1c13f48ef8158bfc9d'), (b'Jefe', b'what do ya want for nothing?', b'750c783e6ab0b503eaa86e310a5db738'), (b'\xaa' * 16, b'\xdd' * 50, b'56be34521d144c88dbb8c733f0e8b3f6')]
        for key, data, mac in vectors:
            outMAC = transport.SSHCiphers(b'none', b'none', b'hmac-md5', b'none')
            outMAC.outMAC = outMAC._getMAC(b'hmac-md5', key)
            seqid, = struct.unpack('>L', data[:4])
            shortened = data[4:]
            self.assertEqual(mac, binascii.hexlify(outMAC.makeMAC(seqid, shortened)), f'Failed HMAC test vector; key={key!r} data={data!r}')