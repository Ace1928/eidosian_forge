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
def _send_service_request(self, path, error_msg_prefix, data=None, **kwargs):
    try:
        self._ignore_http_errors = True
        return self.connection.send(path, data, **kwargs)
    except HTTPError as e:
        error_msg = self._response_to_json(to_text(e.read()))
        raise ConnectionError('%s: %s' % (error_msg_prefix, error_msg), http_code=e.code)
    finally:
        self._ignore_http_errors = False