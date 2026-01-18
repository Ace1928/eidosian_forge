import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def HidUsageSelector(device):
    if device['usage_page'] == 61904 and device['usage'] == 1:
        return True
    return False