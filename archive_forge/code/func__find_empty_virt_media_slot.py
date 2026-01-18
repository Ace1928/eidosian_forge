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
@staticmethod
def _find_empty_virt_media_slot(resources, media_types, media_match_strict=True, vendor=''):
    for uri, data in resources.items():
        if 'MediaTypes' in data and media_types:
            if not set(media_types).intersection(set(data['MediaTypes'])):
                continue
        elif media_match_strict:
            continue
        if vendor == 'Lenovo' and ('RDOC' in uri or 'Remote' in uri):
            continue
        if not data.get('Inserted', False) and (not data.get('ImageName')):
            return (uri, data)
    return (None, None)