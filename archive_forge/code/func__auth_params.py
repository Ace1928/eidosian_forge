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
def _auth_params(self, headers):
    """
        Return tuple of required authentication params based on the presence
        of a token in the self.creds dict. If using a token, set the
        X-Auth-Token header in the `headers` param.

        :param headers: dict containing headers to send in request
        :return: tuple of username, password and force_basic_auth
        """
    if self.creds.get('token'):
        username = None
        password = None
        force_basic_auth = False
        headers['X-Auth-Token'] = self.creds['token']
    else:
        username = self.creds['user']
        password = self.creds['pswd']
        force_basic_auth = True
    return (username, password, force_basic_auth)