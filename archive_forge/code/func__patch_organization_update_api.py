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
def _patch_organization_update_api(self):
    """
        This is a workaround for the broken organization update apidoc in Katello.
        See https://projects.theforeman.org/issues/27538
        """
    _organization_methods = self.foremanapi.apidoc['docs']['resources']['organizations']['methods']
    _organization_update = next((x for x in _organization_methods if x['name'] == 'update'))
    _organization_update_params_organization = next((x for x in _organization_update['params'] if x['name'] == 'organization'))
    _organization_update_params_organization['required'] = False