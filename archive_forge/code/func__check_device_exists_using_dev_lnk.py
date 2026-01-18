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
def _check_device_exists_using_dev_lnk(self, uuid):
    lnk_path = f'/dev/disk/by-id/nvme-uuid.{uuid}'
    if os.path.exists(lnk_path):
        devname = os.path.realpath(lnk_path)
        if devname.startswith('/dev/nvme'):
            LOG.info('LIGHTOS: devpath %s detected for uuid %s', devname, uuid)
            return devname
    return None