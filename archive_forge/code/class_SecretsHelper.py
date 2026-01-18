import argparse
import getpass
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from zunclient import api_versions
from zunclient import client as base_client
from zunclient.common.apiclient import auth
from zunclient.common import cliutils
from zunclient import exceptions as exc
from zunclient.i18n import _
from zunclient.v1 import shell as shell_v1
from zunclient import version
class SecretsHelper(object):

    def __init__(self, args, client):
        self.args = args
        self.client = client
        self.key = None

    def _validate_string(self, text):
        if text is None or len(text) == 0:
            return False
        return True

    def _make_key(self):
        if self.key is not None:
            return self.key
        keys = [self.client.auth_url, self.client.projectid, self.client.user, self.client.region_name, self.client.endpoint_type, self.client.service_type, self.client.service_name, self.client.volume_service_name]
        for index, key in enumerate(keys):
            if key is None:
                keys[index] = '?'
            else:
                keys[index] = str(keys[index])
        self.key = '/'.join(keys)
        return self.key

    def _prompt_password(self, verify=True):
        pw = None
        if hasattr(sys.stdin, 'isatty') and sys.stdin.isatty():
            try:
                while True:
                    pw1 = getpass.getpass('OS Password: ')
                    if verify:
                        pw2 = getpass.getpass('Please verify: ')
                    else:
                        pw2 = pw1
                    if pw1 == pw2 and self._validate_string(pw1):
                        pw = pw1
                        break
            except EOFError:
                pass
        return pw

    def save(self, auth_token, management_url, tenant_id):
        if not HAS_KEYRING or not self.args.os_cache:
            return
        if auth_token == self.auth_token and management_url == self.management_url:
            return
        if not all([management_url, auth_token, tenant_id]):
            raise ValueError('Unable to save empty management url/auth token')
        value = '|'.join([str(auth_token), str(management_url), str(tenant_id)])
        keyring.set_password('zunclient_auth', self._make_key(), value)

    @property
    def password(self):
        if self._validate_string(self.args.os_password):
            return self.args.os_password
        verify_pass = strutils.bool_from_string(cliutils.env('OS_VERIFY_PASSWORD'))
        return self._prompt_password(verify_pass)

    @property
    def management_url(self):
        if not HAS_KEYRING or not self.args.os_cache:
            return None
        management_url = None
        try:
            block = keyring.get_password('zunclient_auth', self._make_key())
            if block:
                _token, management_url, _tenant_id = block.split('|', 2)
        except all_errors:
            pass
        return management_url

    @property
    def auth_token(self):
        if not HAS_KEYRING or not self.args.os_cache:
            return None
        token = None
        try:
            block = keyring.get_password('zunclient_auth', self._make_key())
            if block:
                token, _management_url, _tenant_id = block.split('|', 2)
        except all_errors:
            pass
        return token

    @property
    def tenant_id(self):
        if not HAS_KEYRING or not self.args.os_cache:
            return None
        tenant_id = None
        try:
            block = keyring.get_password('zunclient_auth', self._make_key())
            if block:
                _token, _management_url, tenant_id = block.split('|', 2)
        except all_errors:
            pass
        return tenant_id