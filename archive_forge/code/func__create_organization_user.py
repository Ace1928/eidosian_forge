from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def _create_organization_user(self, org_id, login, role):
    return self._api_call('POST', 'orgs/%d/users' % org_id, {'loginOrEmail': login, 'role': role})