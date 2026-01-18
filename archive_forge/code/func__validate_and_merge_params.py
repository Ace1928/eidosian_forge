from __future__ import (absolute_import, division, print_function)
import random
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible_collections.f5networks.f5_modules.plugins.module_utils.bigiq import F5RestClient
def _validate_and_merge_params(self, **kwargs):
    self.username = kwargs.pop('username', 'admin')
    self.password = kwargs.pop('password', 'admin')
    self.validate_certs = kwargs.pop('validate_certs', False)
    self.host = kwargs.pop('host', None)
    self.port = kwargs.pop('port', 443)
    self.pool_name = kwargs.pop('pool_name', None)
    if self.host is None:
        raise AnsibleError('A valid hostname or IP for BIGIQ needs to be provided')
    if self.pool_name is None:
        raise AnsibleError('License pool name needs to be specified')
    self.params = dict(provider=dict(server=self.host, server_port=self.port, validate_certs=self.validate_certs, user=self.username, password=self.password))