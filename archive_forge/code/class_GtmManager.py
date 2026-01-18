from __future__ import absolute_import, division, print_function
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class GtmManager(BaseManager):

    def exists(self):
        uri = 'https://{0}:{1}/mgmt/tm/gtm/rule/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError:
            return False
        if resp.status == 404 or ('code' in response and response['code'] == 404):
            return False
        return True

    def update_on_device(self):
        params = self.changes.api_params()
        uri = 'https://{0}:{1}/mgmt/tm/gtm/rule/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.patch(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(resp.content)

    def create_on_device(self):
        params = self.changes.api_params()
        params['name'] = self.want.name
        params['partition'] = self.want.partition
        uri = 'https://{0}:{1}/mgmt/tm/gtm/rule/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(resp.content)

    def read_current_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/gtm/rule/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return ApiParameters(params=response)
        raise F5ModuleError(resp.content)

    def remove_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/gtm/rule/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        response = self.client.api.delete(uri)
        if response.status in [200, 201]:
            return True
        raise F5ModuleError(response.content)