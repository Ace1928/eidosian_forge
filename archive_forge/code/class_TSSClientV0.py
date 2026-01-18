from __future__ import absolute_import, division, print_function
import abc
import os
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils import six
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
class TSSClientV0(TSSClient):

    def __init__(self, **server_parameters):
        super(TSSClientV0, self).__init__()
        if server_parameters.get('domain'):
            raise AnsibleError("The 'domain' option requires 'python-tss-sdk' version 1.0.0 or greater")
        self._client = SecretServer(server_parameters['base_url'], server_parameters['username'], server_parameters['password'], server_parameters['api_path_uri'], server_parameters['token_path_uri'])