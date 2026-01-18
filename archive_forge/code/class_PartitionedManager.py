from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import string_types
from ansible.module_utils._text import to_bytes
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
class PartitionedManager(BaseManager):

    def exists(self):
        response = self.list_users_on_device()
        if 'items' in response:
            collection = [x for x in response['items'] if x['name'] == self.want.username_credential]
            if len(collection) == 1:
                return True
            elif len(collection) == 0:
                return False
            else:
                raise F5ModuleError('Multiple users with the provided name were found!')
        return False

    def create_on_device(self):
        params = self.changes.api_params()
        params['name'] = self.want.username_credential
        params['partition'] = self.want.partition
        uri = 'https://{0}:{1}/mgmt/tm/auth/user/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] in [400, 404, 409, 403]:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        return True

    def read_current_from_device(self):
        response = self.list_users_on_device()
        collection = [x for x in response['items'] if x['name'] == self.want.username_credential]
        if len(collection) == 1:
            user = collection.pop()
            return ApiParameters(params=user)
        elif len(collection) == 0:
            raise F5ModuleError('No accounts with the provided name were found.')
        else:
            raise F5ModuleError('Multiple users with the provided name were found!')

    def update_on_device(self):
        params = self.changes.api_params()
        uri = 'https://{0}:{1}/mgmt/tm/auth/user/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], self.want.username_credential)
        resp = self.client.api.patch(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] in [400, 404, 409, 403]:
            if 'message' in response:
                if 'updated successfully' not in response['message']:
                    raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)

    def remove_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/auth/user/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], self.want.username_credential)
        response = self.client.api.delete(uri)
        if response.status == 200:
            return True
        raise F5ModuleError(response.content)

    def list_users_on_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/auth/user/'.format(self.client.provider['server'], self.client.provider['server_port'])
        query = "?$filter=partition+eq+'{0}'".format(self.want.partition)
        resp = self.client.api.get(uri + query)
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