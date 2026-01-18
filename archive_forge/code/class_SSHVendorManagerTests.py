from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
class SSHVendorManagerTests(TestCaseWithTransport):

    def test_register_vendor(self):
        manager = TestSSHVendorManager()
        self.overrideEnv('BRZ_SSH', None)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)
        vendor = object()
        manager.register_vendor('vendor', vendor)
        self.overrideEnv('BRZ_SSH', 'vendor')
        self.assertIs(manager.get_vendor(), vendor)

    def test_default_vendor(self):
        manager = TestSSHVendorManager()
        self.overrideEnv('BRZ_SSH', None)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)
        vendor = object()
        manager.register_default_vendor(vendor)
        self.assertIs(manager.get_vendor(), vendor)

    def test_get_vendor_by_environment(self):
        manager = TestSSHVendorManager()
        self.overrideEnv('BRZ_SSH', None)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)
        self.overrideEnv('BRZ_SSH', 'vendor')
        self.assertRaises(UnknownSSH, manager.get_vendor)
        vendor = object()
        manager.register_vendor('vendor', vendor)
        self.assertIs(manager.get_vendor(), vendor)

    def test_get_vendor_by_config(self):
        manager = TestSSHVendorManager()
        self.overrideEnv('BRZ_SSH', None)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)
        config.GlobalStack().set('ssh', 'vendor')
        self.assertRaises(UnknownSSH, manager.get_vendor)
        vendor = object()
        manager.register_vendor('vendor', vendor)
        self.assertIs(manager.get_vendor(), vendor)

    def test_get_vendor_by_inspection_openssh(self):
        manager = TestSSHVendorManager()
        self.overrideEnv('BRZ_SSH', None)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)
        manager.set_ssh_version_string('OpenSSH')
        self.assertIsInstance(manager.get_vendor(), OpenSSHSubprocessVendor)

    def test_get_vendor_by_inspection_sshcorp(self):
        manager = TestSSHVendorManager()
        self.overrideEnv('BRZ_SSH', None)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)
        manager.set_ssh_version_string('SSH Secure Shell')
        self.assertIsInstance(manager.get_vendor(), SSHCorpSubprocessVendor)

    def test_get_vendor_by_inspection_lsh(self):
        manager = TestSSHVendorManager()
        self.overrideEnv('BRZ_SSH', None)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)
        manager.set_ssh_version_string('lsh')
        self.assertIsInstance(manager.get_vendor(), LSHSubprocessVendor)

    def test_get_vendor_by_inspection_plink(self):
        manager = TestSSHVendorManager()
        self.overrideEnv('BRZ_SSH', None)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)
        manager.set_ssh_version_string('plink')
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)

    def test_cached_vendor(self):
        manager = TestSSHVendorManager()
        self.overrideEnv('BRZ_SSH', None)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)
        vendor = object()
        manager.register_vendor('vendor', vendor)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)
        self.overrideEnv('BRZ_SSH', 'vendor')
        self.assertIs(manager.get_vendor(), vendor)
        self.overrideEnv('BRZ_SSH', None)
        self.assertIs(manager.get_vendor(), vendor)
        manager.clear_cache()
        self.overrideEnv('BRZ_SSH', None)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)

    def test_get_vendor_search_order(self):
        manager = TestSSHVendorManager()
        self.overrideEnv('BRZ_SSH', None)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor)
        default_vendor = object()
        manager.register_default_vendor(default_vendor)
        self.assertIs(manager.get_vendor(), default_vendor)
        manager.clear_cache()
        manager.set_ssh_version_string('OpenSSH')
        self.assertIsInstance(manager.get_vendor(), OpenSSHSubprocessVendor)
        manager.clear_cache()
        vendor = object()
        manager.register_vendor('vendor', vendor)
        self.overrideEnv('BRZ_SSH', 'vendor')
        self.assertIs(manager.get_vendor(), vendor)
        self.overrideEnv('BRZ_SSH', 'vendor')
        self.assertIs(manager.get_vendor(), vendor)

    def test_get_vendor_from_path_win32_plink(self):
        manager = TestSSHVendorManager()
        manager.set_ssh_version_string('plink: Release 0.60')
        plink_path = 'C:/Program Files/PuTTY/plink.exe'
        self.overrideEnv('BRZ_SSH', plink_path)
        vendor = manager.get_vendor()
        self.assertIsInstance(vendor, PLinkSubprocessVendor)
        args = vendor._get_vendor_specific_argv('user', 'host', 22, ['bzr'])
        self.assertEqual(args[0], plink_path)

    def test_get_vendor_from_path_nix_openssh(self):
        manager = TestSSHVendorManager()
        manager.set_ssh_version_string('OpenSSH_5.1p1 Debian-5, OpenSSL, 0.9.8g 19 Oct 2007')
        openssh_path = '/usr/bin/ssh'
        self.overrideEnv('BRZ_SSH', openssh_path)
        vendor = manager.get_vendor()
        self.assertIsInstance(vendor, OpenSSHSubprocessVendor)
        args = vendor._get_vendor_specific_argv('user', 'host', 22, ['bzr'])
        self.assertEqual(args[0], openssh_path)