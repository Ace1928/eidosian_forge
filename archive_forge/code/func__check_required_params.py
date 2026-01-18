from __future__ import (absolute_import, division, print_function)
import abc
import os
import json
import subprocess
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleLookupError, AnsibleOptionsError
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.six import with_metaclass
from ansible_collections.community.general.plugins.module_utils.onepassword import OnePasswordConfig
def _check_required_params(self, required_params):
    non_empty_attrs = dict(((param, getattr(self, param, None)) for param in required_params if getattr(self, param, None)))
    missing = set(required_params).difference(non_empty_attrs)
    if missing:
        prefix = 'Unable to sign in to 1Password. Missing required parameter'
        plural = ''
        suffix = ': {params}.'.format(params=', '.join(missing))
        if len(missing) > 1:
            plural = 's'
        msg = '{prefix}{plural}{suffix}'.format(prefix=prefix, plural=plural, suffix=suffix)
        raise AnsibleLookupError(msg)