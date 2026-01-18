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
def _lookup_login_url(self, payload):
    """ Try to find correct login URL and get api token using this URL.

        :param payload: Token request payload
        :type payload: dict
        :return: token generation response
        """
    preconfigured_token_path = self._get_api_token_path()
    if preconfigured_token_path:
        token_paths = [preconfigured_token_path]
    else:
        token_paths = self._get_known_token_paths()
    for url in token_paths:
        try:
            response = self._send_login_request(payload, url)
        except ConnectionError as e:
            self.connection.queue_message('vvvv', 'REST:request to %s failed because of connection error: %s ' % (url, e))
            if hasattr(e, 'http_code') and e.http_code == 400:
                raise
        else:
            if not preconfigured_token_path:
                self._set_api_token_path(url)
            return response
    raise ConnectionError(INVALID_API_TOKEN_PATH_MSG if preconfigured_token_path else MISSING_API_TOKEN_PATH_MSG)