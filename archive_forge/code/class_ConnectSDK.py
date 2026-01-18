from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
class ConnectSDK:
    primary_url, primary_user, primary_ca = ('', '', '')
    second_url, second_user, second_ca = ('', '', '')
    prefix = ''
    error_msg = "%s%s The '%s' field in the %s setup is not initialized in var file mapping.%s"

    def __init__(self, var_file, primary_pwd, second_pwd):
        """
        ---
        dr_sites_primary_url: http://xxx.xx.xx.xxx:8080/ovirt-engine/api
        dr_sites_primary_username: admin@internal
        dr_sites_primary_ca_file: /etc/pki/ovirt-engine/ca.pem

        # Please fill in the following properties for the secondary site:
        dr_sites_secondary_url: http://yyy.yy.yy.yyy:8080/ovirt-engine/api
        dr_sites_secondary_username: admin@internal
        dr_sites_secondary_ca_file: /etc/pki/ovirt-engine_secondary/ca.pem
        """
        self.primary_url = var_file.get('dr_sites_primary_url')
        self.primary_user = var_file.get('dr_sites_primary_username')
        self.primary_ca = var_file.get('dr_sites_primary_ca_file')
        self.second_url = var_file.get('dr_sites_secondary_url')
        self.second_user = var_file.get('dr_sites_secondary_username')
        self.second_ca = var_file.get('dr_sites_secondary_ca_file')
        self.primary_pwd = primary_pwd
        self.second_pwd = second_pwd

    def validate_primary(self):
        isValid = True
        if self.primary_url is None:
            print(self.error_msg % (FAIL, PREFIX, 'url', 'primary', END))
            isValid = False
        if self.primary_user is None:
            print(self.error_msg % (FAIL, PREFIX, 'username', 'primary', END))
            isValid = False
        if self.primary_ca is None:
            print(self.error_msg % (FAIL, PREFIX, 'ca', 'primary', END))
            isValid = False
        return isValid

    def validate_secondary(self):
        isValid = True
        if self.second_url is None:
            print(self.error_msg % (FAIL, PREFIX, 'url', 'secondary', END))
            isValid = False
        if self.second_user is None:
            print(self.error_msg % (FAIL, PREFIX, 'username', 'secondary', END))
            isValid = False
        if self.second_ca is None:
            print(self.error_msg % (FAIL, PREFIX, 'ca', 'secondary', END))
            isValid = False
        return isValid

    def _validate_connection(self, url, username, password, ca):
        conn = None
        try:
            conn = self._connect_sdk(url, username, password, ca)
            dcs_service = conn.system_service().data_centers_service()
            dcs_service.list()
        except Exception:
            print('%s%sConnection to setup has failed. Please check your credentials: \n%s URL: %s\n%s user: %s\n%s CA file: %s%s' % (FAIL, PREFIX, PREFIX, url, PREFIX, username, PREFIX, ca, END))
            if conn:
                conn.close()
            return None
        return conn

    def connect_primary(self):
        return self._validate_connection(self.primary_url, self.primary_user, self.primary_pwd, self.primary_ca)

    def connect_secondary(self):
        return self._validate_connection(self.second_url, self.second_user, self.second_pwd, self.second_ca)

    def _connect_sdk(self, url, username, password, ca):
        connection = sdk.Connection(url=url, username=username, password=password, ca_file=ca)
        return connection