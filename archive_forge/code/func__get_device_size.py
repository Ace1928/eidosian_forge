import logging
import math
import os
import time
from oslo_config import cfg
from oslo_utils import units
from glance_store._drivers.cinder import base
from glance_store import exceptions
from glance_store.i18n import _
@staticmethod
def _get_device_size(device_file):
    device_file.seek(0, os.SEEK_END)
    device_size = device_file.tell()
    device_size = int(math.ceil(float(device_size) / units.Gi))
    return device_size