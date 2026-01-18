from __future__ import absolute_import, division, print_function
import abc
import os
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils import six
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
class TSSClientV1(TSSClient):

    def __init__(self, **server_parameters):
        super(TSSClientV1, self).__init__()
        authorizer = self._get_authorizer(**server_parameters)
        self._client = SecretServer(server_parameters['base_url'], authorizer, server_parameters['api_path_uri'])

    @staticmethod
    def _get_authorizer(**server_parameters):
        if server_parameters.get('token'):
            return AccessTokenAuthorizer(server_parameters['token'])
        if server_parameters.get('domain'):
            return DomainPasswordGrantAuthorizer(server_parameters['base_url'], server_parameters['username'], server_parameters['domain'], server_parameters['password'], server_parameters['token_path_uri'])
        return PasswordGrantAuthorizer(server_parameters['base_url'], server_parameters['username'], server_parameters['password'], server_parameters['token_path_uri'])