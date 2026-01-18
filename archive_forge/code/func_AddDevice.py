import base64
import os
import sys
import mock
from pyu2f.hid import linux
def AddDevice(fs, dev_name, product_name, vendor_id, product_id, report_descriptor_b64):
    uevent = fs.CreateFile('/sys/class/hidraw/%s/device/uevent' % dev_name)
    rd = fs.CreateFile('/sys/class/hidraw/%s/device/report_descriptor' % dev_name)
    report_descriptor = base64.b64decode(report_descriptor_b64)
    rd.SetContents(report_descriptor)
    buf = 'HID_NAME=%s\n' % product_name.encode('utf8')
    buf += 'HID_ID=0001:%08X:%08X\n' % (vendor_id, product_id)
    uevent.SetContents(buf)