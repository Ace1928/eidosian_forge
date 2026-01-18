import base64
import os
import sys
import mock
from pyu2f.hid import linux
class LinuxTest(unittest.TestCase):

    def setUp(self):
        self.fs = fake_filesystem.FakeFilesystem()
        self.fs.CreateDirectory('/sys/class/hidraw')

    def tearDown(self):
        self.fs.RemoveObject('/sys/class/hidraw')

    def testCallEnumerate(self):
        AddDevice(self.fs, 'hidraw1', 'Logitech USB Keyboard', 1133, 49948, KEYBOARD_RD)
        AddDevice(self.fs, 'hidraw2', 'Yubico U2F', 4176, 1031, YUBICO_RD)
        with mock.patch.object(linux, 'os', fake_filesystem.FakeOsModule(self.fs)):
            fake_open = fake_filesystem.FakeFileOpen(self.fs)
            with mock.patch.object(py_builtins, 'open', fake_open):
                devs = list(linux.LinuxHidDevice.Enumerate())
                devs = sorted(devs, key=lambda k: k['vendor_id'])
                self.assertEquals(len(devs), 2)
                self.assertEquals(devs[0]['vendor_id'], 1133)
                self.assertEquals(devs[0]['product_id'], 49948)
                self.assertEquals(devs[1]['vendor_id'], 4176)
                self.assertEquals(devs[1]['product_id'], 1031)
                self.assertEquals(devs[1]['usage_page'], 61904)
                self.assertEquals(devs[1]['usage'], 1)

    def testCallOpen(self):
        AddDevice(self.fs, 'hidraw1', 'Yubico U2F', 4176, 1031, YUBICO_RD)
        fake_open = fake_filesystem.FakeFileOpen(self.fs)
        with mock.patch.object(py_builtins, 'open', fake_open):
            fake_dev_os = FakeDeviceOsModule()
            with mock.patch.object(linux, 'os', fake_dev_os):
                dev = linux.LinuxHidDevice('/dev/hidraw1')
                self.assertEquals(dev.GetInReportDataLength(), 64)
                self.assertEquals(dev.GetOutReportDataLength(), 64)
                dev.Write(list(range(0, 64)))
                self.assertEquals(list(fake_dev_os.data_written), [0] + list(range(0, 64)))
                fake_dev_os.data_to_return = b'x' * 64
                self.assertEquals(dev.Read(), [120] * 64)