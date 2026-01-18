from __future__ import (absolute_import, division, print_function)
import os
import socket
import ssl
import json
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.common.text.converters import to_text
def _raise_err_from_json(self, resp_json):
    err_params = {}
    if self.debug:
        err_params['logs'] = self.logs
    raise LXDClientException(self._get_err_from_resp_json(resp_json), **err_params)