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
class LicenseFactManager(BaseManager):

    def __init__(self, *args, **kwargs):
        self.client = kwargs.get('client', None)
        self.module = kwargs.get('module', None)
        super(LicenseFactManager, self).__init__(**kwargs)

    def exec_module(self):
        facts = self._exec_module()
        result = dict(license=facts)
        return result

    def _exec_module(self):
        facts = self.read_facts()
        result = facts.to_return()
        return result

    def read_facts(self):
        resource = self.read_collection_from_device()
        params = LicenseParameters(params=resource)
        return params

    def read_collection_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/sys/license/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(resp.content)
        result = dict()
        try:
            result['license'] = response['entries']['https://localhost/mgmt/tm/sys/license/0']['nestedStats']['entries']
            return result
        except KeyError:
            return None