from __future__ import (absolute_import, division, print_function)
import re
import time
import traceback
from ansible.module_utils.basic import (AnsibleModule, env_fallback,
from ansible.module_utils.common.text.converters import to_text
class _ServiceClient(object):

    def __init__(self, client, endpoint, product):
        self._client = client
        self._endpoint = endpoint
        self._default_header = {'User-Agent': 'Huawei-Ansible-MM-%s' % product, 'Accept': 'application/json'}

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, e):
        self._endpoint = e

    @session_method_wrapper
    def get(self, url, body=None, header=None, timeout=None):
        return self._client.get(url, json=body, timeout=timeout, headers=self._header(header))

    @session_method_wrapper
    def post(self, url, body=None, header=None, timeout=None):
        return self._client.post(url, json=body, timeout=timeout, headers=self._header(header))

    @session_method_wrapper
    def delete(self, url, body=None, header=None, timeout=None):
        return self._client.delete(url, json=body, timeout=timeout, headers=self._header(header))

    @session_method_wrapper
    def put(self, url, body=None, header=None, timeout=None):
        return self._client.put(url, json=body, timeout=timeout, headers=self._header(header))

    def _header(self, header):
        if header and isinstance(header, dict):
            for k, v in self._default_header.items():
                if k not in header:
                    header[k] = v
        else:
            header = self._default_header
        return header