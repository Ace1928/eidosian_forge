from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
class V2Manager(BaseManager):

    def read_current_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/sys/ucs'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        return response

    def read_current(self):
        collection = self.read_current_from_device()
        if 'items' not in collection:
            return []
        resources = collection['items']
        result = [x['apiRawValues']['filename'] for x in resources]
        return result

    def exists(self):
        collection = self.read_current()
        base = os.path.basename(self.want.src)
        if any((base == os.path.basename(x) for x in collection)):
            return True
        return False

    def download_from_device(self, dest):
        url = 'https://{0}:{1}/mgmt/shared/file-transfer/ucs-downloads/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], self.want.src)
        try:
            download_file(self.client, url, dest)
        except F5ModuleError:
            raise F5ModuleError('Failed to download the file.')
        if os.path.exists(self.want.dest):
            return True
        return False