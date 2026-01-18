from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def _construct_url_1(self, obj, config_only=True):
    """
        This method is used by construct_url when the object is the top-level class.
        """
    obj_class = obj.get('aci_class')
    obj_rn = obj.get('aci_rn')
    obj_filter = obj.get('target_filter')
    mo = obj.get('module_object')
    if self.module.params.get('state') in ('absent', 'present'):
        self.path = 'api/mo/uni/{0}.json'.format(obj_rn)
        self.parent_path = 'api/mo/uni.json'
        if config_only:
            self.update_qs({'rsp-prop-include': 'config-only'})
        self.obj_filter = obj_filter
    elif mo is None:
        self.path = 'api/class/{0}.json'.format(obj_class)
        if obj_filter is not None:
            self.update_qs({'query-target-filter': self.build_filter(obj_class, obj_filter)})
    else:
        self.path = 'api/mo/uni/{0}.json'.format(obj_rn)