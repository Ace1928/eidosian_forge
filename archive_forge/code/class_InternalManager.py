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
class InternalManager(BaseManager):

    def create(self):
        self._set_changed_options()
        if size_exceeded(self.want.records_src) or lines_exceeded(self.want.records_src):
            raise F5ModuleError('The size of the provided data (or file) is too large for an internal data group.')
        if self.module.check_mode:
            return True
        self.create_on_device()
        return True

    def update(self):
        self.have = self.read_current_from_device()
        if not self.should_update():
            return False
        if self.module.check_mode:
            return True
        self.update_on_device()
        return True

    def exists(self):
        errors = [401, 403, 409, 500, 501, 502, 503, 504]
        uri = 'https://{0}:{1}/mgmt/tm/ltm/data-group/internal/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status == 404 or ('code' in response and response['code'] == 404):
            return False
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        if resp.status in errors or ('code' in response and response['code'] in errors):
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)

    def create_on_device(self):
        params = self.changes.api_params()
        params['name'] = self.want.name
        params['partition'] = self.want.partition
        uri = 'https://{0}:{1}/mgmt/tm/ltm/data-group/internal/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(resp.content)

    def update_on_device(self):
        params = self.changes.api_params()
        uri = 'https://{0}:{1}/mgmt/tm/ltm/data-group/internal/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.patch(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(resp.content)

    def remove_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/data-group/internal/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        response = self.client.api.delete(uri)
        if response.status in [200, 201]:
            return True
        raise F5ModuleError(response.content)

    def read_current_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/data-group/internal/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return ApiParameters(params=response)
        raise F5ModuleError(resp.content)