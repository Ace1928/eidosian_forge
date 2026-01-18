import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
class ConchTestServerFactory(factory.SSHFactory):
    noisy = False
    services = {b'ssh-userauth': userauth.SSHUserAuthServer, b'ssh-connection': connection.SSHConnection}

    def buildProtocol(self, addr):
        proto = ConchTestServer()
        proto.supportedPublicKeys = list(chain.from_iterable((key.supportedSignatureAlgorithms() for key in self.privateKeys.values())))
        proto.factory = self
        if hasattr(self, 'expectedLoseConnection'):
            proto.expectedLoseConnection = self.expectedLoseConnection
        self.proto = proto
        return proto

    def getPublicKeys(self):
        return {b'ssh-rsa': keys.Key.fromString(publicRSA_openssh), b'ssh-dss': keys.Key.fromString(publicDSA_openssh)}

    def getPrivateKeys(self):
        return {b'ssh-rsa': keys.Key.fromString(privateRSA_openssh), b'ssh-dss': keys.Key.fromString(privateDSA_openssh)}

    def getPrimes(self) -> Dict[int, List[Tuple[int, int]]]:
        """
            Diffie-Hellman primes that can be used for the
            diffie-hellman-group-exchange-sha1 key exchange.

            @return: The primes and generators.
            @rtype: L{dict} mapping the key size to a C{list} of
                C{(generator, prime)} tupple.
            """
        return {2048: [_kex.getDHGeneratorAndPrime(b'diffie-hellman-group14-sha1')]}

    def getService(self, trans, name):
        return factory.SSHFactory.getService(self, trans, name)