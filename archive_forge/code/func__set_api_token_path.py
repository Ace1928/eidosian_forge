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
def _set_api_token_path(self, url):
    return self.set_option(API_TOKEN_PATH_OPTION_NAME, url)