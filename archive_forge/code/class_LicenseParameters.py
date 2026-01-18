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
class LicenseParameters(BaseParameters):
    api_map = {}
    returnables = ['license_start_date', 'license_end_date', 'licensed_on_date', 'licensed_version', 'max_permitted_version', 'min_permitted_version', 'platform_id', 'registration_key', 'service_check_date', 'active_modules']

    @property
    def license_start_date(self):
        if self._values['license'] is None:
            return None
        if bool(self._values['license'].get('licenseStartDate')):
            return self._values['license']['licenseStartDate']['description']

    @property
    def license_end_date(self):
        if self._values['license'] is None:
            return None
        if bool(self._values['license'].get('licenseEndDate')):
            return self._values['license']['licenseEndDate']['description']

    @property
    def licensed_on_date(self):
        if self._values['license'] is None:
            return None
        if bool(self._values['license'].get('licensedOnDate')):
            return self._values['license']['licensedOnDate']['description']

    @property
    def licensed_version(self):
        if self._values['license'] is None:
            return None
        if bool(self._values['license'].get('licensedVersion')):
            return self._values['license']['licensedVersion']['description']

    @property
    def max_permitted_version(self):
        if self._values['license'] is None:
            return None
        if bool(self._values['license'].get('maxPermittedVersion')):
            return self._values['license']['maxPermittedVersion']['description']

    @property
    def min_permitted_version(self):
        if self._values['license'] is None:
            return None
        if bool(self._values['license'].get('minPermittedVersion')):
            return self._values['license']['minPermittedVersion']['description']

    @property
    def platform_id(self):
        if self._values['license'] is None:
            return None
        if bool(self._values['license'].get('platformId')):
            return self._values['license']['platformId']['description']

    @property
    def registration_key(self):
        if self._values['license'] is None:
            return None
        if bool(self._values['license'].get('registrationKey')):
            return self._values['license']['registrationKey']['description']

    @property
    def service_check_date(self):
        if self._values['license'] is None:
            return None
        if bool(self._values['license'].get('serviceCheckDate')):
            return self._values['license']['serviceCheckDate']['description']

    @property
    def active_modules(self):
        if self._values['license'] is None:
            return None
        result = list()
        license = self._values['license']
        for key in license:
            if key.startswith('http'):
                v = license[key]['nestedStats']['entries']
                for k in v.keys():
                    addons = {k2: v2['description'] for k2, v2 in v[k]['nestedStats']['entries'].items()}
                    result.append(addons)
        return result