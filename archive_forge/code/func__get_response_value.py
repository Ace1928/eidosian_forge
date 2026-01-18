from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils.basic import to_text
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible_collections.ansible.netcommon.plugins.plugin_utils.httpapi_base import HttpApiBase
def _get_response_value(self, response_data):
    return to_text(response_data.getvalue())