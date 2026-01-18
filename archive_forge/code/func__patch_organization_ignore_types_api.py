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
@_check_patch_needed(fixed_version='3.5.0', plugins=['katello'])
def _patch_organization_ignore_types_api(self):
    """
        This is a workaround for the missing ignore_types in the organization apidoc in Katello.
        See https://projects.theforeman.org/issues/35687
        """
    _ignore_types_param = {'name': 'ignore_types', 'full_name': 'organization[ignore_types]', 'description': '\n<p>List of resources types that will be automatically associated</p>\n', 'required': False, 'allow_nil': True, 'allow_blank': False, 'validator': 'Must be an array of any type', 'expected_type': 'array', 'metadata': None, 'show': True, 'validations': []}
    _organization_methods = self.foremanapi.apidoc['docs']['resources']['organizations']['methods']
    _organization_create = next((x for x in _organization_methods if x['name'] == 'create'))
    _organization_update = next((x for x in _organization_methods if x['name'] == 'update'))
    if next((x for x in _organization_create['params'] if x['name'] == 'ignore_types'), None) is None:
        _organization_create['params'].append(_ignore_types_param)
        _organization_update['params'].append(_ignore_types_param)