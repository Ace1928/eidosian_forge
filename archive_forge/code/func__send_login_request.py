from __future__ import (absolute_import, division, print_function)
import json
import os
import re
from ansible import __version__ as ansible_version
from ansible.module_utils.basic import to_text
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import FdmSwaggerParser, SpecProp, FdmSwaggerValidator
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, ResponseParams
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.connection import ConnectionError
def _send_login_request(self, payload, url):
    self._display(HTTPMethod.POST, 'login', url)
    response, response_data = self._send_auth_request(url, json.dumps(payload), method=HTTPMethod.POST, headers=BASE_HEADERS)
    self._display(HTTPMethod.POST, 'login:status_code', response.getcode())
    response = self._response_to_json(self._get_response_value(response_data))
    return response