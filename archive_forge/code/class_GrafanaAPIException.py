from __future__ import absolute_import, division, print_function
import json
import os
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.urls import basic_auth_header, open_url
from ansible.module_utils._text import to_native
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.utils.display import Display
class GrafanaAPIException(Exception):
    pass