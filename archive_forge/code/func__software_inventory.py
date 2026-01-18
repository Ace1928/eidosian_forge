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
def _software_inventory(self, uri):
    result = {}
    result['entries'] = []
    while uri:
        response = self.get_request(self.root_uri + uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        if data.get('Members@odata.nextLink'):
            uri = data.get('Members@odata.nextLink')
        else:
            uri = None
        for member in data[u'Members']:
            fw_uri = self.root_uri + member[u'@odata.id']
            response = self.get_request(fw_uri)
            if response['ret'] is False:
                return response
            result['ret'] = True
            data = response['data']
            software = {}
            for key in ['Name', 'Id', 'Status', 'Version', 'Updateable', 'SoftwareId', 'LowestSupportedVersion', 'Manufacturer', 'ReleaseDate']:
                if key in data:
                    software[key] = data.get(key)
            result['entries'].append(software)
    return result