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
@_check_patch_needed(introduced_version='2.1.0', fixed_version='2.3.0')
def _patch_subnet_externalipam_group_api(self):
    """
        This is a workaround for the broken subnet apidoc for External IPAM.
        See https://projects.theforeman.org/issues/30890
        """
    _subnet_externalipam_group_parameter = {u'validations': [], u'name': u'externalipam_group', u'show': True, u'description': u'\n<p>External IPAM group - only relevant when IPAM is set to external</p>\n', u'required': False, u'allow_nil': True, u'allow_blank': False, u'full_name': u'subnet[externalipam_group]', u'expected_type': u'string', u'metadata': None, u'validator': u''}
    _subnet_methods = self.foremanapi.apidoc['docs']['resources']['subnets']['methods']
    _subnet_create = next((x for x in _subnet_methods if x['name'] == 'create'))
    _subnet_create_params_subnet = next((x for x in _subnet_create['params'] if x['name'] == 'subnet'))
    _subnet_create_params_subnet['params'].append(_subnet_externalipam_group_parameter)
    _subnet_update = next((x for x in _subnet_methods if x['name'] == 'update'))
    _subnet_update_params_subnet = next((x for x in _subnet_update['params'] if x['name'] == 'subnet'))
    _subnet_update_params_subnet['params'].append(_subnet_externalipam_group_parameter)