from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip_interface
from ..module_utils.teem import send_teem
def _upload_to_file(self, name, type, remote_path, update=False):
    self.upload_file_to_device(self.want.records_src, name)
    if update:
        uri = 'https://{0}:{1}/mgmt/tm/sys/file/data-group/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, name))
        params = {'sourcePath': 'file:{0}'.format(remote_path)}
        resp = self.client.api.patch(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
    else:
        uri = 'https://{0}:{1}/mgmt/tm/sys/file/data-group/'.format(self.client.provider['server'], self.client.provider['server_port'])
        params = dict(name=name, type=type, sourcePath='file:{0}'.format(remote_path), partition=self.want.partition)
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        return response['name']
    raise F5ModuleError(resp.content)