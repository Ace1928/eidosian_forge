from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import fetch_url
def a10_argument_spec():
    return dict(host=dict(type='str', required=True), username=dict(type='str', aliases=['user', 'admin'], required=True), password=dict(type='str', aliases=['pass', 'pwd'], required=True, no_log=True), write_config=dict(type='bool', default=False))