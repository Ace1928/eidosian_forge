from __future__ import (absolute_import, division, print_function)
import os
import socket
import ssl
import json
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.common.text.converters import to_text
@staticmethod
def _get_err_from_resp_json(resp_json):
    err = None
    metadata = resp_json.get('metadata', None)
    if metadata is not None:
        err = metadata.get('err', None)
    if err is None:
        err = resp_json.get('error', None)
    return err