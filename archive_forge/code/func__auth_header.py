from __future__ import absolute_import, division, print_function
import datetime
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def _auth_header(self):
    return 'Token token=%s' % self.token