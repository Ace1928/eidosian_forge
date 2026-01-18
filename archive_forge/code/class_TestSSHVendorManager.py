from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
class TestSSHVendorManager(SSHVendorManager):
    _ssh_version_string = ''

    def set_ssh_version_string(self, version):
        self._ssh_version_string = version

    def _get_ssh_version_string(self, args):
        return self._ssh_version_string