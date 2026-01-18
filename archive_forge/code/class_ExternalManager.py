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
class ExternalManager(BaseManager):

    def absent(self):
        result = False
        if self.exists():
            result = self.remove()
        if self.external_file_exists() and self.want.delete_data_group_file:
            result = self.remove_data_group_file_from_device()
        return result

    def create(self):
        if zero_length(self.want.records_src):
            raise F5ModuleError('An external data group cannot be empty.')
        self._set_changed_options()
        if self.module.check_mode:
            return True
        self.create_on_device()
        return True

    def update(self):
        self.have = self.read_current_from_device()
        if not self.should_update():
            return False
        if self.changes.records_src and zero_length(self.want.records_src):
            raise F5ModuleError('An external data group cannot be empty.')
        if self.module.check_mode:
            return True
        self.update_on_device()
        return True

    def exists(self):
        errors = [401, 403, 409, 500, 501, 502, 503, 504]
        uri = 'https://{0}:{1}/mgmt/tm/ltm/data-group/external/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
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

    def external_file_exists(self):
        errors = [401, 403, 409, 500, 501, 502, 503, 504]
        uri = 'https://{0}:{1}/mgmt/tm/sys/file/data-group/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.external_file_name))
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

    def upload_file_to_device(self, content, name):
        url = 'https://{0}:{1}/mgmt/shared/file-transfer/uploads'.format(self.client.provider['server'], self.client.provider['server_port'])
        try:
            upload_file(self.client, url, content, name)
        except F5ModuleError:
            raise F5ModuleError('Failed to upload the file.')

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

    def remove_file_on_device(self, remote_path):
        uri = 'https://{0}:{1}/mgmt/tm/util/unix-rm/'.format(self.client.provider['server'], self.client.provider['server_port'])
        args = dict(command='run', utilCmdArgs=remote_path)
        resp = self.client.api.post(uri, json=args)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(response.content)

    def create_on_device(self):
        name = self.want.external_file_name
        remote_path = '/var/config/rest/downloads/{0}'.format(name)
        external_file = self._upload_to_file(name, self.want.type, remote_path, update=False)
        params = dict(name=self.want.name, partition=self.want.partition, externalFileName=external_file)
        if self.want.description:
            params['description'] = self.want.description
        uri = 'https://{0}:{1}/mgmt/tm/ltm/data-group/external/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(response.content)
        self.remove_file_on_device(remote_path)

    def update_on_device(self):
        params = {}
        if self.want.records_src is not None:
            name = self.want.external_file_name
            remote_path = '/var/config/rest/downloads/{0}'.format(name)
            external_file = self._upload_to_file(name, self.have.type, remote_path, update=True)
            params['externalFileName'] = external_file
        if self.changes.description is not None:
            params['description'] = self.changes.description
        if not params:
            return
        uri = 'https://{0}:{1}/mgmt/tm/ltm/data-group/external/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp = self.client.api.patch(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(resp.content)

    def remove_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/data-group/external/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        response = self.client.api.delete(uri)
        if self.want.delete_data_group_file:
            self.remove_data_group_file_from_device()
        if response.status in [200, 201]:
            return True
        raise F5ModuleError(response.content)

    def remove_data_group_file_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/sys/file/data-group/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.external_file_name))
        response = self.client.api.delete(uri)
        if response.status in [200, 201]:
            return True
        raise F5ModuleError(response.content)

    def read_current_from_device(self):
        """Reads the current configuration from the device

        For an external data group, we are interested in two things from the
        current configuration

        * ``checksum``
        * ``type``

        The ``checksum`` will allow us to compare the data group value we have
        with the data group value being provided.

        The ``type`` will allow us to do validation on the data group value being
        provided (if any).

        Returns:
             ExternalApiParameters: Attributes of the remote resource.
        """
        uri = 'https://{0}:{1}/mgmt/tm/ltm/data-group/external/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
        resp_dg = self.client.api.get(uri)
        try:
            response_dg = resp_dg.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp_dg.status not in [200, 201] or ('code' in response_dg and response_dg['code'] not in [200, 201]):
            raise F5ModuleError(resp_dg.content)
        external_file = os.path.basename(response_dg['externalFileName'])
        external_file_partition = os.path.dirname(response_dg['externalFileName']).strip('/')
        uri = 'https://{0}:{1}/mgmt/tm/sys/file/data-group/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(external_file_partition, external_file))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            result = ApiParameters(params=response)
            result.update({'description': response_dg.get('description', None)})
            return result
        raise F5ModuleError(resp.content)