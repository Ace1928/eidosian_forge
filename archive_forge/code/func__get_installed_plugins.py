from __future__ import absolute_import, division, print_function
import hashlib
import io
import json
import os
import tempfile
from ansible.module_utils.basic import AnsibleModule, to_bytes
from ansible.module_utils.six.moves import http_cookiejar as cookiejar
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url, url_argument_spec
from ansible.module_utils.six import text_type, binary_type
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.jenkins import download_updates_file
def _get_installed_plugins(self):
    plugins_data = self._get_json_data('%s/%s' % (self.url, 'pluginManager/api/json?depth=1'), 'list of plugins')
    if 'plugins' not in plugins_data:
        self.module.fail_json(msg='No valid plugin data found.')
    self.is_installed = False
    self.is_pinned = False
    self.is_enabled = False
    for p in plugins_data['plugins']:
        if p['shortName'] == self.params['name']:
            self.is_installed = True
            if p['pinned']:
                self.is_pinned = True
            if p['enabled']:
                self.is_enabled = True
            break