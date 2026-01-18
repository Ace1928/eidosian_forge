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
class AsmPolicyFactManager(BaseManager):

    def __init__(self, *args, **kwargs):
        self.client = kwargs.get('client', None)
        self.module = kwargs.get('module', None)
        super(AsmPolicyFactManager, self).__init__(**kwargs)

    def exec_module(self):
        facts = self._exec_module()
        result = dict(asm_policies=facts)
        return result

    def _exec_module(self):
        if 'asm' not in self.provisioned_modules:
            return []
        manager = self.get_manager()
        return manager._exec_module()

    def get_manager(self):
        if self.version_is_less_than_13():
            return AsmPolicyFactManagerV12(**self.kwargs)
        else:
            return AsmPolicyFactManagerV13(**self.kwargs)

    def version_is_less_than_13(self):
        version = tmos_version(self.client)
        if Version(version) < Version('13.0.0'):
            return True
        else:
            return False

    def read_facts(self):
        results = []
        collection = self.increment_read()
        for resource in collection:
            params = AsmPolicyFactParameters(params=resource)
            results.append(params)
        return results

    def increment_read(self):
        n = 0
        result = []
        while True:
            items = self.read_collection_from_device(skip=n)
            if not items:
                break
            result.extend(items)
            n = n + 10
        return result