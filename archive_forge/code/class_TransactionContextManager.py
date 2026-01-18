from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.urls import urlparse
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.urls import Request
from .common import F5ModuleError
from ansible.module_utils.six.moves.urllib.error import HTTPError
from .constants import (
class TransactionContextManager(object):

    def __init__(self, client, validate_only=False):
        self.client = client
        self.validate_only = validate_only
        self.transid = None

    def __enter__(self):
        uri = 'https://{0}:{1}/mgmt/tm/transaction/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json={})
        if resp.status not in [200]:
            raise Exception
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        self.transid = response['transId']
        self.client.api.request.headers['X-F5-REST-Coordination-Id'] = self.transid
        return self.client

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.client.api.request.headers.pop('X-F5-REST-Coordination-Id')
        if exc_tb is None:
            uri = 'https://{0}:{1}/mgmt/tm/transaction/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], self.transid)
            params = dict(state='VALIDATING', validateOnly=self.validate_only)
            resp = self.client.api.patch(uri, json=params)
            if resp.status not in [200]:
                raise Exception