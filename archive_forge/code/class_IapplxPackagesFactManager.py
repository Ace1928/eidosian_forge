from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class IapplxPackagesFactManager(BaseManager):

    def __init__(self, *args, **kwargs):
        self.client = kwargs.get('client', None)
        self.module = kwargs.get('module', None)
        super(IapplxPackagesFactManager, self).__init__(**kwargs)

    def exec_module(self):
        facts = self._exec_module()
        result = dict(iapplx_packages=facts)
        return result

    def _exec_module(self):
        results = []
        facts = self.read_facts()
        for item in facts:
            attrs = item.to_return()
            results.append(attrs)
        results = sorted(results, key=lambda k: k['name'])
        return results

    def read_facts(self):
        results = []
        collection = self.read_collection_from_device()
        for resource in collection:
            params = IapplxPackagesParameters(params=resource)
            results.append(params)
        return results

    def read_collection_from_device(self):
        params = dict(operation='QUERY')
        uri = 'https://{0}:{1}/mgmt/shared/iapp/package-management-tasks'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201, 202] or ('code' in response and response['code'] not in [200, 201, 202]):
            raise F5ModuleError(resp.content)
        status = self.wait_for_task(response['id'])
        if status == 'FINISHED':
            uri = 'https://{0}:{1}/mgmt/shared/iapp/package-management-tasks/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], response['id'])
            resp = self.client.api.get(uri)
            try:
                response = resp.json()
            except ValueError as ex:
                raise F5ModuleError(str(ex))
            if resp.status not in [200, 201, 202] or ('code' in response and response['code'] not in [200, 201, 202]):
                raise F5ModuleError(resp.content)
        else:
            raise F5ModuleError('An error occurred querying iAppLX packages.')
        result = response['queryResponse']
        return result

    def wait_for_task(self, task_id):
        uri = 'https://{0}:{1}/mgmt/shared/iapp/package-management-tasks/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], task_id)
        for x in range(0, 60):
            resp = self.client.api.get(uri)
            try:
                response = resp.json()
            except ValueError as ex:
                raise F5ModuleError(str(ex))
            if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
                raise F5ModuleError(resp.content)
            if response['status'] in ['FINISHED', 'FAILED']:
                return response['status']
            time.sleep(1)
        return response['status']