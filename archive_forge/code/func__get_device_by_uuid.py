import glob
import http.client
import os
import re
import tempfile
import time
import traceback
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import lightos as priv_lightos
from os_brick import utils
@utils.trace
def _get_device_by_uuid(self, uuid):
    endtime = time.time() + self.WAIT_DEVICE_TIMEOUT
    while time.time() < endtime:
        try:
            device = self._check_device_exists_using_dev_lnk(uuid)
            if device:
                return device
        except Exception as e:
            LOG.debug('LIGHTOS: %s', e)
        device = self._check_device_exists_reading_block_class(uuid)
        if device:
            return device
        time.sleep(1)
    return None