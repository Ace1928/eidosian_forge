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
def _find_sessionservice_resource(self):
    response = self.get_request(self.root_uri + self.service_root)
    if response['ret'] is False:
        return response
    data = response['data']
    self.session_service_uri = data.get('SessionService', {}).get('@odata.id')
    self.sessions_uri = data.get('Links', {}).get('Sessions', {}).get('@odata.id')
    if self.session_service_uri is None:
        return {'ret': False, 'msg': 'SessionService resource not found'}
    if self.sessions_uri is None:
        return {'ret': False, 'msg': 'SessionCollection resource not found'}
    return {'ret': True}