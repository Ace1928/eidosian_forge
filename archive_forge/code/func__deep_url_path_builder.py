from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def _deep_url_path_builder(self, obj):
    target_class = obj.get('target_class')
    target_filter = obj.get('target_filter')
    subtree_class = obj.get('subtree_class')
    subtree_filter = obj.get('subtree_filter')
    object_rn = obj.get('object_rn')
    mo = obj.get('module_object')
    add_subtree_filter = obj.get('add_subtree_filter')
    add_target_filter = obj.get('add_target_filter')
    if self.module.params.get('state') in ('absent', 'present') and mo is not None:
        self.path = 'api/mo/uni/{0}.json'.format(object_rn)
        self.update_qs({'rsp-prop-include': 'config-only'})
    else:
        if object_rn is not None:
            self.path = 'api/mo/uni/{0}.json'.format(object_rn)
        else:
            self.path = 'api/class/{0}.json'.format(target_class)
        if add_target_filter:
            self.update_qs({'query-target-filter': self.build_filter(target_class, target_filter)})
        if add_subtree_filter:
            self.update_qs({'rsp-subtree-filter': self.build_filter(subtree_class, subtree_filter)})
    if self.params.get('port') is not None:
        self.url = '{protocol}://{host}:{port}/{path}'.format(path=self.path, **self.module.params)
    else:
        self.url = '{protocol}://{host}/{path}'.format(path=self.path, **self.module.params)
    if self.child_classes:
        self.update_qs({'rsp-subtree': 'full', 'rsp-subtree-class': ','.join(sorted(self.child_classes))})