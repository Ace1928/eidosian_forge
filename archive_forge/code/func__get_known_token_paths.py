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
def _get_known_token_paths(self):
    """Generate list of token generation urls based on list of versions supported by device(if exposed via API) or
        default list of API versions.

        :returns: list of token generation urls
        :rtype: generator
        """
    try:
        api_versions = self._get_supported_api_versions()
    except ConnectionError:
        api_versions = DEFAULT_API_VERSIONS
    return [TOKEN_PATH_TEMPLATE.format(version) for version in api_versions]