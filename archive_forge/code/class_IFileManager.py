from __future__ import absolute_import, division, print_function
import hashlib
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
class IFileManager(BaseManager):

    def create_on_device(self):
        params = self.changes.api_params()
        params['name'] = self.want.name
        params['source-path'] = 'file:/var/config/rest/downloads/{0}'.format(self.want.name)
        params['partition'] = self.want.partition
        uri = 'https://{0}:{1}/mgmt/tm/sys/file/ifile/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(resp.content)

    def exists(self):
        uri = 'https://{0}:{1}/mgmt/tm/sys/file/ifile/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status == 404 or ('code' in response and response['code'] == 404):
            return False
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        errors = [401, 403, 409, 500, 501, 502, 503, 504]
        if resp.status in errors or ('code' in response and response['code'] in errors):
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)

    def read_current_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/sys/file/ifile/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return ApiParameters(params=response)
        raise F5ModuleError(resp.content)

    def remove_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/sys/file/ifile/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        response = self.client.api.delete(uri)
        if response.status == 200:
            return True
        raise F5ModuleError(response.content)