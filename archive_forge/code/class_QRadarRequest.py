from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.six.moves.urllib.parse import quote_plus
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from copy import copy
import json
class QRadarRequest(object):

    def __init__(self, module=None, connection=None, headers=None, not_rest_data_keys=None, task_vars=None):
        self.module = module
        if module:
            self.connection = Connection(self.module._socket_path)
        elif connection:
            self.connection = connection
            try:
                self.connection.load_platform_plugins('ibm.qradar.qradar')
                self.connection.set_options(var_options=task_vars)
            except ConnectionError:
                raise
        if not_rest_data_keys:
            self.not_rest_data_keys = not_rest_data_keys
        else:
            self.not_rest_data_keys = []
        self.not_rest_data_keys.append('validate_certs')
        self.headers = headers if headers else BASE_HEADERS

    def _httpapi_error_handle(self, method, uri, payload=None):
        code = 99999
        response = {}
        try:
            code, response = self.connection.send_request(method, uri, payload=payload, headers=self.headers)
        except ConnectionError as e:
            self.module.fail_json(msg='connection error occurred: {0}'.format(e))
        except CertificateError as e:
            self.module.fail_json(msg='certificate error occurred: {0}'.format(e))
        except ValueError as e:
            try:
                self.module.fail_json(msg='certificate not found: {0}'.format(e))
            except AttributeError:
                pass
        if code == 404:
            if to_text('Object not found') in to_text(response) or to_text('Could not find object') in to_text(response) or to_text('No offense was found') in to_text(response):
                return {}
            if to_text('The rule does not exist.') in to_text(response['description']):
                return (code, {})
        if code == 409:
            if 'code' in response:
                if response['code'] in [1002, 1004]:
                    return response
                else:
                    self.module.fail_json(msg='qradar httpapi returned error {0} with message {1}'.format(code, response))
        elif not (code >= 200 and code < 300):
            try:
                self.module.fail_json(msg='qradar httpapi returned error {0} with message {1}'.format(code, response))
            except AttributeError:
                pass
        return (code, response)

    def get(self, url, **kwargs):
        return self._httpapi_error_handle('GET', url, **kwargs)

    def put(self, url, **kwargs):
        return self._httpapi_error_handle('PUT', url, **kwargs)

    def post(self, url, **kwargs):
        return self._httpapi_error_handle('POST', url, **kwargs)

    def patch(self, url, **kwargs):
        return self._httpapi_error_handle('PATCH', url, **kwargs)

    def delete(self, url, **kwargs):
        return self._httpapi_error_handle('DELETE', url, **kwargs)

    def get_data(self):
        """
        Get the valid fields that should be passed to the REST API as urlencoded
        data so long as the argument specification to the module follows the
        convention:
            - the key to the argspec item does not start with qradar_
            - the key does not exist in the not_data_keys list
        """
        try:
            qradar_data = {}
            for param in self.module.params:
                if self.module.params[param] is not None and param not in self.not_rest_data_keys:
                    qradar_data[param] = self.module.params[param]
            return qradar_data
        except TypeError as e:
            self.module.fail_json(msg='invalid data type provided: {0}'.format(e))

    def post_by_path(self, rest_path, data=None):
        """
        POST with data to path
        """
        if data is None:
            data = json.dumps(self.get_data())
        elif data is False:
            return self.post('/{0}'.format(rest_path))
        return self.post('/{0}'.format(rest_path), payload=data)

    def create_update(self, rest_path, data=None):
        """
        Create or Update a file/directory monitor data input in qradar
        """
        if data is None:
            data = json.dumps(self.get_data())
        return self.patch('/{0}'.format(rest_path), payload=data)