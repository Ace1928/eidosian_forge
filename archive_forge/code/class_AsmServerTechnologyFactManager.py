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
class AsmServerTechnologyFactManager(BaseManager):

    def __init__(self, *args, **kwargs):
        self.client = kwargs.get('client', None)
        self.module = kwargs.get('module', None)
        super(AsmServerTechnologyFactManager, self).__init__(**kwargs)

    def exec_module(self):
        facts = self._exec_module()
        result = dict(asm_server_technologies=facts)
        return result

    def _exec_module(self):
        results = []
        if 'asm' not in self.provisioned_modules:
            return results
        if self.version_is_less_than_13():
            return results
        facts = self.read_facts()
        for item in facts:
            attrs = item.to_return()
            results.append(attrs)
        results = sorted(results, key=lambda k: k['server_technology_name'])
        return results

    def version_is_less_than_13(self):
        version = tmos_version(self.client)
        if Version(version) < Version('13.0.0'):
            return True
        else:
            return False

    def read_facts(self):
        results = []
        collection = self.read_collection_from_device()
        for resource in collection:
            params = AsmServerTechnologyFactParameters(params=resource)
            results.append(params)
        return results

    def read_collection_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/asm/server-technologies'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(resp.content)
        if 'items' not in response:
            return []
        result = response['items']
        return result