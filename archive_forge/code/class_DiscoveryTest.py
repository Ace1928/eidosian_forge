from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
class DiscoveryTest(unittest.TestCase):

    def testHidUsageSelector(self):
        u2f_device = {'usage_page': 61904, 'usage': 1}
        other_device = {'usage_page': 6, 'usage': 1}
        self.assertTrue(hidtransport.HidUsageSelector(u2f_device))
        self.assertFalse(hidtransport.HidUsageSelector(other_device))

    def testDiscoverLocalDevices(self):

        def FakeHidDevice(path):
            mock_hid_dev = mock.MagicMock()
            mock_hid_dev.GetInReportDataLength.return_value = 64
            mock_hid_dev.GetOutReportDataLength.return_value = 64
            mock_hid_dev.path = path
            return mock_hid_dev
        with mock.patch.object(hidtransport, 'hid') as hid_mock:
            with mock.patch.object(hidtransport.UsbHidTransport, 'InternalInit') as _:
                hid_mock.Enumerate.return_value = [MakeKeyboard('path1', 6), MakeKeyboard('path2', 2), MakeU2F('path3'), MakeU2F('path4')]
                mock_hid_dev = mock.MagicMock()
                mock_hid_dev.GetInReportDataLength.return_value = 64
                mock_hid_dev.GetOutReportDataLength.return_value = 64
                hid_mock.Open.side_effect = FakeHidDevice
                devs = list(hidtransport.DiscoverLocalHIDU2FDevices())
                self.assertEquals(hid_mock.Enumerate.call_count, 1)
                self.assertEquals(hid_mock.Open.call_count, 2)
                self.assertEquals(len(devs), 2)
                self.assertEquals(devs[0].hid_device.path, 'path3')
                self.assertEquals(devs[1].hid_device.path, 'path4')