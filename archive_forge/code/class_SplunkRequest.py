from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlencode
class SplunkRequest(object):

    def __init__(self, module=None, headers=None, action_module=None, connection=None, keymap=None, not_rest_data_keys=None, override=True):
        self.module = module
        if module:
            self.connection = Connection(self.module._socket_path)
            self.legacy = True
        elif connection:
            self.connection = connection
            try:
                self.connection.load_platform_plugins('splunk.es.splunk')
                self.module = action_module
                self.legacy = False
            except ConnectionError:
                raise
        if keymap is None:
            self.keymap = {}
        else:
            self.keymap = keymap
        self.override = override
        if not_rest_data_keys is None:
            self.not_rest_data_keys = []
        else:
            self.not_rest_data_keys = not_rest_data_keys
        self.not_rest_data_keys.append('validate_certs')

    def _httpapi_error_handle(self, method, uri, payload=None):
        try:
            code, response = self.connection.send_request(method, uri, payload=payload)
            if code == 404:
                if to_text('Object not found') in to_text(response) or to_text('Could not find object') in to_text(response):
                    return {}
            if not (code >= 200 and code < 300):
                self.module.fail_json(msg='Splunk httpapi returned error {0} with message {1}'.format(code, response))
            return response
        except ConnectionError as e:
            self.module.fail_json(msg='connection error occurred: {0}'.format(e))
        except CertificateError as e:
            self.module.fail_json(msg='certificate error occurred: {0}'.format(e))
        except ValueError as e:
            try:
                self.module.fail_json(msg='certificate not found: {0}'.format(e))
            except AttributeError:
                pass

    def get(self, url, **kwargs):
        return self._httpapi_error_handle('GET', url, **kwargs)

    def put(self, url, **kwargs):
        return self._httpapi_error_handle('PUT', url, **kwargs)

    def post(self, url, **kwargs):
        return self._httpapi_error_handle('POST', url, **kwargs)

    def delete(self, url, **kwargs):
        return self._httpapi_error_handle('DELETE', url, **kwargs)

    def get_data(self, config=None):
        """
        Get the valid fields that should be passed to the REST API as urlencoded
        data so long as the argument specification to the module follows the
        convention:
            - the key to the argspec item does not start with splunk_
            - the key does not exist in the not_data_keys list
        """
        try:
            splunk_data = {}
            if self.legacy and (not config):
                config = self.module.params
            for param in config:
                if config[param] is not None and param not in self.not_rest_data_keys:
                    if param in self.keymap:
                        splunk_data[self.keymap[param]] = config[param]
                    else:
                        splunk_data[param] = config[param]
            return splunk_data
        except TypeError as e:
            self.module.fail_json(msg='invalid data type provided: {0}'.format(e))

    def get_urlencoded_data(self, config):
        return urlencode(self.get_data(config))

    def get_by_path(self, rest_path):
        """
        GET attributes of a monitor by rest path
        """
        return self.get('/{0}?output_mode=json'.format(rest_path))

    def delete_by_path(self, rest_path):
        """
        DELETE attributes of a monitor by rest path
        """
        return self.delete('/{0}?output_mode=json'.format(rest_path))

    def create_update(self, rest_path, data):
        """
        Create or Update a file/directory monitor data input in Splunk
        """
        if data is not None and self.override:
            data = self.get_urlencoded_data(data)
        return self.post('/{0}?output_mode=json'.format(rest_path), payload=data)