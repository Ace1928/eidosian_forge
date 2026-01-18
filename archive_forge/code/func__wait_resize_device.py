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
def _wait_resize_device(volume, device_file):
    timeout = 20
    max_recheck_wait = 10
    tries = 0
    elapsed = 0
    while ScaleIOBrickConnector._get_device_size(device_file) < volume.size:
        wait = min(0.5 * 2 ** tries, max_recheck_wait)
        time.sleep(wait)
        tries += 1
        elapsed += wait
        if elapsed >= timeout:
            msg = _('Timeout while waiting while volume %(volume_id)s to resize the device in %(tries)s tries.') % {'volume_id': volume.id, 'tries': tries}
            LOG.error(msg)
            raise exceptions.BackendException(msg)