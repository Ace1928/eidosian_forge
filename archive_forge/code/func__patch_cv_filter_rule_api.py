from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
@_check_patch_needed(plugins=['katello'])
def _patch_cv_filter_rule_api(self):
    """
        This is a workaround for missing params of CV Filter Rule update controller in Katello.
        See https://projects.theforeman.org/issues/30908
        """
    _content_view_filter_rule_methods = self.foremanapi.apidoc['docs']['resources']['content_view_filter_rules']['methods']
    _content_view_filter_rule_create = next((x for x in _content_view_filter_rule_methods if x['name'] == 'create'))
    _content_view_filter_rule_update = next((x for x in _content_view_filter_rule_methods if x['name'] == 'update'))
    for param_name in ['uuid', 'errata_ids', 'date_type', 'module_stream_ids']:
        create_param = next((x for x in _content_view_filter_rule_create['params'] if x['name'] == param_name), None)
        update_param = next((x for x in _content_view_filter_rule_update['params'] if x['name'] == param_name), None)
        if create_param is not None and update_param is None:
            _content_view_filter_rule_update['params'].append(create_param)