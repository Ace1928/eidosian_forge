from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _entity_validator(self, python_vars):
    ovirt_setups = ConnectSDK(python_vars, self.primary_pwd, self.second_pwd)
    isValid = ovirt_setups.validate_primary()
    isValid = ovirt_setups.validate_secondary() and isValid
    if isValid:
        primary_conn, second_conn = ('', '')
        try:
            primary_conn = ovirt_setups.connect_primary()
            if primary_conn is None:
                return False
            isValid = self._validate_entities_in_setup(primary_conn, 'primary', python_vars) and isValid
            second_conn = ovirt_setups.connect_secondary()
            if second_conn is None:
                return False
            isValid = self._validate_entities_in_setup(second_conn, 'secondary', python_vars) and isValid
            cluster_mapping = python_vars.get(self.cluster_map)
            isValid = isValid and self._validate_vms_for_failback(primary_conn, 'primary')
            isValid = isValid and self._validate_vms_for_failback(second_conn, 'secondary')
            isValid = isValid and self._is_compatible_versions(primary_conn, second_conn, cluster_mapping)
        finally:
            if primary_conn:
                primary_conn.close()
            if second_conn:
                second_conn.close()
    return isValid