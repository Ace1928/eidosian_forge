import json
import os
import urllib
from oslo_log import log as logging
import requests
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.privileged import scaleio as priv_scaleio
from os_brick import utils
@utils.retry(exception.BrickException, retries=15, backoff_rate=1)
def _wait_for_volume_path(self, path):
    if not os.path.isdir(path):
        msg = _('ScaleIO volume %(volume_id)s not found at expected path.') % {'volume_id': self.volume_id}
        LOG.debug(msg)
        raise exception.BrickException(message=msg)
    disk_filename = None
    filenames = os.listdir(path)
    LOG.info('Files found in %(path)s path: %(files)s ', {'path': path, 'files': filenames})
    for filename in filenames:
        if filename.startswith('emc-vol') and filename.endswith(self.volume_id):
            disk_filename = filename
            break
    if not disk_filename:
        msg = _('ScaleIO volume %(volume_id)s not found.') % {'volume_id': self.volume_id}
        LOG.debug(msg)
        raise exception.BrickException(message=msg)
    return disk_filename