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
class ManagedDevicesParameters(BaseParameters):
    api_map = {'deviceUri': 'device_uri', 'groupName': 'group_name', 'httpsPort': 'https_port', 'isClustered': 'is_clustered', 'isLicenseExpired': 'is_license_expired', 'isVirtual': 'is_virtual', 'machineId': 'machine_id', 'managementAddress': 'management_address', 'mcpDeviceName': 'mcp_device_name', 'restFrameworkVersion': 'rest_framework_version', 'selfLink': 'self_link', 'trustDomainGuid': 'trust_domain_guid'}
    returnables = ['address', 'build', 'device_uri', 'edition', 'group_name', 'hostname', 'https_port', 'is_clustered', 'is_license_expired', 'is_virtual', 'machine_id', 'management_address', 'mcp_device_name', 'product', 'rest_framework_version', 'self_link', 'slots', 'state', 'tags', 'trust_domain_guid', 'uuid', 'version']

    @property
    def slots(self):
        result = []
        if self._values['slots'] is None:
            return None
        for x in self._values['slots']:
            x['is_active'] = flatten_boolean(x.pop('isActive', False))
            result.append(x)
        return result

    @property
    def tags(self):
        if self._values['tags'] is None:
            return None
        result = dict(((x['name'], x['value']) for x in self._values['tags']))
        return result

    @property
    def https_port(self):
        return int(self._values['https_port'])

    @property
    def is_clustered(self):
        return flatten_boolean(self._values['is_clustered'])

    @property
    def is_license_expired(self):
        return flatten_boolean(self._values['is_license_expired'])

    @property
    def is_virtual(self):
        return flatten_boolean(self._values['is_virtual'])