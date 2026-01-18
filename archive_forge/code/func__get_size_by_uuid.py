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
def _get_size_by_uuid(self, uuid):
    devpath = self._get_device_by_uuid(uuid)
    devname = devpath.split('/')[-1]
    try:
        size_path_name = os.path.join('/sys/class/block/', devname, 'size')
        with open(size_path_name, 'r') as f:
            size_blks = f.read().strip()
        bytesize = int(size_blks) * 512
        return bytesize
    except Exception:
        LOG.warning('LIGHTOS: Could not find the size at for uuid %s in %s', uuid, devpath)
        return None