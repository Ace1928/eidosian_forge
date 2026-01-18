from __future__ import absolute_import, division, print_function
import json
import os
import re
import traceback
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import Request
class ECSSession(object):

    def __init__(self, name, **kwargs):
        """
        Initialize our session
        """
        self._set_config(name, **kwargs)

    def client(self):
        resource = Resource(self)
        return resource

    def _set_config(self, name, **kwargs):
        headers = {'Content-Type': 'application/json', 'Connection': 'keep-alive'}
        self.request = Request(headers=headers, timeout=60)
        configurators = [self._read_config_vars]
        for configurator in configurators:
            self._config = configurator(name, **kwargs)
            if self._config:
                break
        if self._config is None:
            raise SessionConfigurationException(to_native('No Configuration Found.'))
        entrust_api_user = self.get_config('entrust_api_user')
        entrust_api_key = self.get_config('entrust_api_key')
        if entrust_api_user and entrust_api_key:
            self.request.url_username = entrust_api_user
            self.request.url_password = entrust_api_key
        else:
            raise SessionConfigurationException(to_native('User and key must be provided.'))
        entrust_api_cert = self.get_config('entrust_api_cert')
        entrust_api_cert_key = self.get_config('entrust_api_cert_key')
        if entrust_api_cert:
            self.request.client_cert = entrust_api_cert
            if entrust_api_cert_key:
                self.request.client_key = entrust_api_cert_key
        else:
            raise SessionConfigurationException(to_native('Client certificate for authentication to the API must be provided.'))
        entrust_api_specification_path = self.get_config('entrust_api_specification_path')
        if not entrust_api_specification_path.startswith('http') and (not os.path.isfile(entrust_api_specification_path)):
            raise SessionConfigurationException(to_native('OpenAPI specification was not found at location {0}.'.format(entrust_api_specification_path)))
        if not valid_file_format.match(entrust_api_specification_path):
            raise SessionConfigurationException(to_native('OpenAPI specification filename must end in .json, .yml or .yaml'))
        self.verify = True
        if entrust_api_specification_path.startswith('http'):
            try:
                http_response = Request().open(method='GET', url=entrust_api_specification_path)
                http_response_contents = http_response.read()
                if entrust_api_specification_path.endswith('.json'):
                    self._spec = json.load(http_response_contents)
                elif entrust_api_specification_path.endswith('.yml') or entrust_api_specification_path.endswith('.yaml'):
                    self._spec = yaml.safe_load(http_response_contents)
            except HTTPError as e:
                raise SessionConfigurationException(to_native("Error downloading specification from address '{0}', received error code '{1}'".format(entrust_api_specification_path, e.getcode())))
        else:
            with open(entrust_api_specification_path) as f:
                if '.json' in entrust_api_specification_path:
                    self._spec = json.load(f)
                elif '.yml' in entrust_api_specification_path or '.yaml' in entrust_api_specification_path:
                    self._spec = yaml.safe_load(f)

    def get_config(self, item):
        return self._config.get(item, None)

    def _read_config_vars(self, name, **kwargs):
        """ Read configuration from variables passed to the module. """
        config = {}
        entrust_api_specification_path = kwargs.get('entrust_api_specification_path')
        if not entrust_api_specification_path or (not entrust_api_specification_path.startswith('http') and (not os.path.isfile(entrust_api_specification_path))):
            raise SessionConfigurationException(to_native("Parameter provided for entrust_api_specification_path of value '{0}' was not a valid file path or HTTPS address.".format(entrust_api_specification_path)))
        for required_file in ['entrust_api_cert', 'entrust_api_cert_key']:
            file_path = kwargs.get(required_file)
            if not file_path or not os.path.isfile(file_path):
                raise SessionConfigurationException(to_native("Parameter provided for {0} of value '{1}' was not a valid file path.".format(required_file, file_path)))
        for required_var in ['entrust_api_user', 'entrust_api_key']:
            if not kwargs.get(required_var):
                raise SessionConfigurationException(to_native('Parameter provided for {0} was missing.'.format(required_var)))
        config['entrust_api_cert'] = kwargs.get('entrust_api_cert')
        config['entrust_api_cert_key'] = kwargs.get('entrust_api_cert_key')
        config['entrust_api_specification_path'] = kwargs.get('entrust_api_specification_path')
        config['entrust_api_user'] = kwargs.get('entrust_api_user')
        config['entrust_api_key'] = kwargs.get('entrust_api_key')
        return config