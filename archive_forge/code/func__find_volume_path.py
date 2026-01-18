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
def _find_volume_path(self):
    LOG.info('Looking for volume %(volume_id)s, maximum tries: %(tries)s', {'volume_id': self.volume_id, 'tries': self.device_scan_attempts})
    by_id_path = self.get_search_path()
    disk_filename = self._wait_for_volume_path(by_id_path)
    full_disk_name = '%(path)s/%(filename)s' % {'path': by_id_path, 'filename': disk_filename}
    LOG.info('Full disk name is %(full_path)s', {'full_path': full_disk_name})
    return full_disk_name