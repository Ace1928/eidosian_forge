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
class PurchasedPoolLicensesParameters(BaseParameters):
    api_map = {'baseRegKey': 'base_reg_key', 'freeDeviceLicenses': 'free_device_licenses', 'licenseState': 'license_state', 'totalDeviceLicenses': 'total_device_licenses'}
    returnables = ['base_reg_key', 'dossier', 'free_device_licenses', 'name', 'state', 'total_device_licenses', 'uuid', 'vendor', 'licensed_date_time', 'licensed_version', 'evaluation_start_date_time', 'evaluation_end_date_time', 'license_end_date_time', 'license_start_date_time', 'registration_key']

    @property
    def registration_key(self):
        try:
            return self._values['license_state']['registrationKey']
        except KeyError:
            return None

    @property
    def license_start_date_time(self):
        try:
            return self._values['license_state']['licenseStartDateTime']
        except KeyError:
            return None

    @property
    def license_end_date_time(self):
        try:
            return self._values['license_state']['licenseEndDateTime']
        except KeyError:
            return None

    @property
    def evaluation_end_date_time(self):
        try:
            return self._values['license_state']['evaluationEndDateTime']
        except KeyError:
            return None

    @property
    def evaluation_start_date_time(self):
        try:
            return self._values['license_state']['evaluationStartDateTime']
        except KeyError:
            return None

    @property
    def licensed_version(self):
        try:
            return self._values['license_state']['licensedVersion']
        except KeyError:
            return None

    @property
    def licensed_date_time(self):
        try:
            return self._values['license_state']['licensedDateTime']
        except KeyError:
            return None

    @property
    def vendor(self):
        try:
            return self._values['license_state']['vendor']
        except KeyError:
            return None