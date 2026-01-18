from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
class SubprocessVendorsTests(TestCase):

    def test_openssh_command_tricked(self):
        vendor = OpenSSHSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', '-oProxyCommand=blah', 100, command=['bzr']), ['ssh', '-oForwardX11=no', '-oForwardAgent=no', '-oClearAllForwardings=yes', '-oNoHostAuthenticationForLocalhost=yes', '-p', '100', '-l', 'user', '--', '-oProxyCommand=blah', 'bzr'])

    def test_openssh_command_arguments(self):
        vendor = OpenSSHSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, command=['bzr']), ['ssh', '-oForwardX11=no', '-oForwardAgent=no', '-oClearAllForwardings=yes', '-oNoHostAuthenticationForLocalhost=yes', '-p', '100', '-l', 'user', '--', 'host', 'bzr'])

    def test_openssh_subsystem_arguments(self):
        vendor = OpenSSHSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, subsystem='sftp'), ['ssh', '-oForwardX11=no', '-oForwardAgent=no', '-oClearAllForwardings=yes', '-oNoHostAuthenticationForLocalhost=yes', '-p', '100', '-l', 'user', '-s', '--', 'host', 'sftp'])

    def test_openssh_command_strange_hostname(self):
        vendor = SSHCorpSubprocessVendor()
        self.assertRaises(StrangeHostname, vendor._get_vendor_specific_argv, 'user', '-oProxyCommand=host', 100, command=['bzr'])

    def test_sshcorp_command_arguments(self):
        vendor = SSHCorpSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, command=['bzr']), ['ssh', '-x', '-p', '100', '-l', 'user', 'host', 'bzr'])

    def test_sshcorp_subsystem_arguments(self):
        vendor = SSHCorpSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, subsystem='sftp'), ['ssh', '-x', '-p', '100', '-l', 'user', '-s', 'sftp', 'host'])

    def test_lsh_command_tricked(self):
        vendor = LSHSubprocessVendor()
        self.assertRaises(StrangeHostname, vendor._get_vendor_specific_argv, 'user', '-oProxyCommand=host', 100, command=['bzr'])

    def test_lsh_command_arguments(self):
        vendor = LSHSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, command=['bzr']), ['lsh', '-p', '100', '-l', 'user', 'host', 'bzr'])

    def test_lsh_subsystem_arguments(self):
        vendor = LSHSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, subsystem='sftp'), ['lsh', '-p', '100', '-l', 'user', '--subsystem', 'sftp', 'host'])

    def test_plink_command_tricked(self):
        vendor = PLinkSubprocessVendor()
        self.assertRaises(StrangeHostname, vendor._get_vendor_specific_argv, 'user', '-oProxyCommand=host', 100, command=['bzr'])

    def test_plink_command_arguments(self):
        vendor = PLinkSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, command=['bzr']), ['plink', '-x', '-a', '-ssh', '-2', '-batch', '-P', '100', '-l', 'user', 'host', 'bzr'])

    def test_plink_subsystem_arguments(self):
        vendor = PLinkSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, subsystem='sftp'), ['plink', '-x', '-a', '-ssh', '-2', '-batch', '-P', '100', '-l', 'user', '-s', 'host', 'sftp'])