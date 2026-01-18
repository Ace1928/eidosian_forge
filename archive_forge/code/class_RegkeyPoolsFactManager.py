from __future__ import absolute_import, division, print_function
import copy
import datetime
import traceback
import math
import re
from ansible.module_utils.basic import (
from ansible.module_utils.six import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
class RegkeyPoolsFactManager(BaseManager):

    def __init__(self, *args, **kwargs):
        self.client = kwargs.get('client', None)
        self.module = kwargs.get('module', None)
        super(RegkeyPoolsFactManager, self).__init__(**kwargs)

    def exec_module(self):
        facts = self._exec_module()
        result = dict(regkey_pools=facts)
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
            params = RegkeyPoolsParameters(params=resource)
            offerings = self.read_offerings_from_device(resource['id'])
            params.update({'total_offerings': len(offerings)})
            for offering in offerings:
                params2 = RegkeyPoolsOfferingParameters(params=offering)
                params.update({'offerings': params2.to_return()})
            results.append(params)
        return results

    def read_collection_from_device(self):
        uri = 'https://{0}:{1}/mgmt/cm/device/licensing/pool/regkey/licenses'.format(self.client.provider['server'], self.client.provider['server_port'])
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
        try:
            return response['items']
        except KeyError:
            return []

    def read_offerings_from_device(self, license):
        uri = 'https://{0}:{1}/mgmt/cm/device/licensing/pool/regkey/licenses/{2}/offerings'.format(self.client.provider['server'], self.client.provider['server_port'], license)
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
        try:
            return response['items']
        except KeyError:
            return []