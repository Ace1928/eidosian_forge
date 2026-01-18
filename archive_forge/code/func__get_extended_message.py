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
def _get_extended_message(error):
    """
        Get Redfish ExtendedInfo message from response payload if present
        :param error: an HTTPError exception
        :type error: HTTPError
        :return: the ExtendedInfo message if present, else standard HTTP error
        """
    msg = http_client.responses.get(error.code, '')
    if error.code >= 400:
        try:
            body = error.read().decode('utf-8')
            data = json.loads(body)
            ext_info = data['error']['@Message.ExtendedInfo']
            try:
                msg = ext_info[0]['Message']
            except Exception:
                msg = str(data['error']['@Message.ExtendedInfo'])
        except Exception:
            pass
    return msg