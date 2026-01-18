from unittest import mock
import ddt
from os_brick import initiator
from os_brick.initiator import connector
from os_brick.initiator.windows import fibre_channel
from os_brick.initiator.windows import iscsi
from os_brick.initiator.windows import smbfs
from os_brick.tests.windows import test_base
@ddt.ddt
class WindowsConnectorFactoryTestCase(test_base.WindowsConnectorTestBase):

    @ddt.data({'proto': initiator.ISCSI, 'expected_cls': iscsi.WindowsISCSIConnector}, {'proto': initiator.FIBRE_CHANNEL, 'expected_cls': fibre_channel.WindowsFCConnector}, {'proto': initiator.SMBFS, 'expected_cls': smbfs.WindowsSMBFSConnector})
    @ddt.unpack
    @mock.patch('sys.platform', 'win32')
    def test_factory(self, proto, expected_cls):
        obj = connector.InitiatorConnector.factory(proto, None)
        self.assertIsInstance(obj, expected_cls)