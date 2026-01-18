import os
import socket
import subprocess
import sys
from itertools import count
from unittest import skipIf
from zope.interface import implementer
from twisted.conch.error import ConchError
from twisted.conch.test.keydata import (
from twisted.conch.test.test_ssh import ConchTestRealm
from twisted.cred import portal
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessExitedAlready
from twisted.internet.task import LoopingCall
from twisted.internet.utils import getProcessValue
from twisted.python import filepath, log, runtime
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
import sys, os
from twisted.conch.scripts.%s import run
class OpenSSHKeyExchangeTests(ConchServerSetupMixin, OpenSSHClientMixin, TestCase):
    """
    Tests L{SSHTransportBase}'s key exchange algorithm compatibility with
    OpenSSH.
    """

    def assertExecuteWithKexAlgorithm(self, keyExchangeAlgo):
        """
        Call execute() method of L{OpenSSHClientMixin} with an ssh option that
        forces the exclusive use of the key exchange algorithm specified by
        keyExchangeAlgo

        @type keyExchangeAlgo: L{str}
        @param keyExchangeAlgo: The key exchange algorithm to use

        @return: L{defer.Deferred}
        """
        kexAlgorithms = []
        try:
            output = subprocess.check_output([which('ssh')[0], '-Q', 'kex'], stderr=subprocess.STDOUT)
            if not isinstance(output, str):
                output = output.decode('utf-8')
            kexAlgorithms = output.split()
        except BaseException:
            pass
        if keyExchangeAlgo not in kexAlgorithms:
            raise SkipTest(f'{keyExchangeAlgo} not supported by ssh client')
        d = self.execute('echo hello', ConchTestOpenSSHProcess(), '-oKexAlgorithms=' + keyExchangeAlgo)
        return d.addCallback(self.assertEqual, b'hello\n')

    def test_ECDHSHA256(self):
        """
        The ecdh-sha2-nistp256 key exchange algorithm is compatible with
        OpenSSH
        """
        return self.assertExecuteWithKexAlgorithm('ecdh-sha2-nistp256')

    def test_ECDHSHA384(self):
        """
        The ecdh-sha2-nistp384 key exchange algorithm is compatible with
        OpenSSH
        """
        return self.assertExecuteWithKexAlgorithm('ecdh-sha2-nistp384')

    def test_ECDHSHA521(self):
        """
        The ecdh-sha2-nistp521 key exchange algorithm is compatible with
        OpenSSH
        """
        return self.assertExecuteWithKexAlgorithm('ecdh-sha2-nistp521')

    def test_DH_GROUP14(self):
        """
        The diffie-hellman-group14-sha1 key exchange algorithm is compatible
        with OpenSSH.
        """
        return self.assertExecuteWithKexAlgorithm('diffie-hellman-group14-sha1')

    def test_DH_GROUP_EXCHANGE_SHA1(self):
        """
        The diffie-hellman-group-exchange-sha1 key exchange algorithm is
        compatible with OpenSSH.
        """
        return self.assertExecuteWithKexAlgorithm('diffie-hellman-group-exchange-sha1')

    def test_DH_GROUP_EXCHANGE_SHA256(self):
        """
        The diffie-hellman-group-exchange-sha256 key exchange algorithm is
        compatible with OpenSSH.
        """
        return self.assertExecuteWithKexAlgorithm('diffie-hellman-group-exchange-sha256')

    def test_unsupported_algorithm(self):
        """
        The list of key exchange algorithms supported
        by OpenSSH client is obtained with C{ssh -Q kex}.
        """
        self.assertRaises(SkipTest, self.assertExecuteWithKexAlgorithm, 'unsupported-algorithm')