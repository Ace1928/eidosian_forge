from __future__ import absolute_import, division, print_function
import json
import os
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
import ssl
def _is_rest(self, used_unsupported_rest_properties=None):
    if self.use_rest == 'Always':
        if used_unsupported_rest_properties:
            error = "REST API currently does not support '%s'" % ', '.join(used_unsupported_rest_properties)
            return (True, error)
        else:
            return (True, None)
    if self.use_rest == 'Never' or used_unsupported_rest_properties:
        return (False, None)
    method = 'HEAD'
    api = 'cluster/software'
    status_code, __ = self.send_request(method, api, params=None, return_status_code=True)
    if status_code == 200:
        return (True, None)
    return (False, None)