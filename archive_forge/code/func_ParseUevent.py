from __future__ import division
import os
import struct
from pyu2f import errors
from pyu2f.hid import base
def ParseUevent(uevent, desc):
    lines = uevent.split(b'\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        k, v = line.split(b'=')
        if k == b'HID_NAME':
            desc.product_string = v.decode('utf8')
        elif k == b'HID_ID':
            _, vid, pid = v.split(b':')
            desc.vendor_id = int(vid, 16)
            desc.product_id = int(pid, 16)