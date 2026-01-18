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
@_check_patch_needed(fixed_version='3.8.0', plugins=['katello'])
def _patch_products_repositories_allow_nil_credential(self):
    """
        This is a workaround for the missing allow_nil: true in the Products and Repositories controllers
        See https://projects.theforeman.org/issues/36497
        """
    for resource in ['products', 'repositories']:
        methods = self.foremanapi.apidoc['docs']['resources'][resource]['methods']
        for action in ['create', 'update']:
            resource_action = next((x for x in methods if x['name'] == action))
            for param in ['gpg_key_id', 'ssl_ca_cert_id', 'ssl_client_cert_id', 'ssl_client_key_id']:
                resource_param = next((x for x in resource_action['params'] if x['name'] == param))
                resource_param['allow_nil'] = True