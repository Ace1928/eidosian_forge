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
def _get_all_action_info_values(self, action):
    """Retrieve all parameter values for an Action from ActionInfo.
        Fall back to AllowableValue annotations if no ActionInfo found.
        Return the result in an ActionInfo-like dictionary, keyed
        by the name of the parameter. """
    ai = {}
    if '@Redfish.ActionInfo' in action:
        ai_uri = action['@Redfish.ActionInfo']
        response = self.get_request(self.root_uri + ai_uri)
        if response['ret'] is True:
            data = response['data']
            if 'Parameters' in data:
                params = data['Parameters']
                ai = dict(((p['Name'], p) for p in params if 'Name' in p))
    if not ai:
        ai = dict(((k[:-24], {'AllowableValues': v}) for k, v in action.items() if k.endswith('@Redfish.AllowableValues')))
    return ai