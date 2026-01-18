from __future__ import absolute_import, division, print_function
from uuid import uuid4
from ssl import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible.module_utils._text import to_text
class ZabbixApiRequest(object):

    def __init__(self, module):
        self.module = module
        self.connection = Connection(self.module._socket_path)

    def _httpapi_error_handle(self, payload=None):
        try:
            code, response = self.connection.send_request(data=payload)
        except ConnectionError as e:
            self.module.fail_json(msg='connection error occurred: {0}'.format(e))
        except CertificateError as e:
            self.module.fail_json(msg='certificate error occurred: {0}'.format(e))
        except ValueError as e:
            self.module.fail_json(msg='certificate not found: {0}'.format(e))
        if code == 404:
            if to_text(u'Object not found') in to_text(response) or to_text(u'Could not find object') in to_text(response):
                return {}
        if not (code >= 200 and code < 300):
            self.module.fail_json(msg='Zabbix httpapi returned error {0} with message {1}'.format(code, response))
        return response

    def api_version(self):
        return self.connection.api_version()

    @staticmethod
    def payload_builder(method_, params, jsonrpc_version='2.0', reqid=str(uuid4()), **kwargs):
        req = {'jsonrpc': jsonrpc_version, 'method': method_, 'id': reqid}
        req['params'] = params
        return req

    def __getattr__(self, name):
        return ZabbixApiSection(self, name)