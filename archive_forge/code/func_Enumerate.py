from __future__ import division
import os
import struct
from pyu2f import errors
from pyu2f.hid import base
@staticmethod
def Enumerate():
    for hidraw in os.listdir('/sys/class/hidraw'):
        rd_path = os.path.join('/sys/class/hidraw', hidraw, 'device/report_descriptor')
        uevent_path = os.path.join('/sys/class/hidraw', hidraw, 'device/uevent')
        rd_file = open(rd_path, 'rb')
        uevent_file = open(uevent_path, 'rb')
        desc = base.DeviceDescriptor()
        desc.path = os.path.join('/dev/', hidraw)
        ParseReportDescriptor(rd_file.read(), desc)
        ParseUevent(uevent_file.read(), desc)
        rd_file.close()
        uevent_file.close()
        yield desc.ToPublicDict()