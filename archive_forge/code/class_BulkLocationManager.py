from __future__ import absolute_import, division, print_function
import os
import re
import socket
import ssl
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
class BulkLocationManager(BaseManager):

    def __init__(self, *args, **kwargs):
        super(BulkLocationManager, self).__init__(**kwargs)
        self.remote_dir = '/var/config/rest/bulk'

    def _download_file(self):
        uri = 'https://{0}:{1}/mgmt/shared/file-transfer/bulk/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], self.want.filename)
        download_file(self.client, uri, self.want.dest)
        if os.path.exists(self.want.dest):
            return True
        return False