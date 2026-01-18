from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _find_empty_account_slot(self):
    response = self.get_request(self.root_uri + self.accounts_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    uris = [a.get('@odata.id') for a in data.get('Members', []) if a.get('@odata.id')]
    if uris:
        uris += [uris.pop(0)]
    for uri in uris:
        response = self.get_request(self.root_uri + uri)
        if response['ret'] is False:
            continue
        data = response['data']
        headers = response['headers']
        if data.get('UserName') == '' and (not data.get('Enabled', True)):
            return {'ret': True, 'data': data, 'headers': headers, 'uri': uri}
    return {'ret': False, 'no_match': True, 'msg': 'No empty account slot found'}