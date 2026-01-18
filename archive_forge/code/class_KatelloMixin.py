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
class KatelloMixin:
    """
    Katello Mixin to extend a :class:`ForemanAnsibleModule` (or any subclass) to work with Katello entities.

    This includes:

    * add a required ``organization`` parameter to the module
    * add Katello to the list of required plugins
    """

    def __init__(self, **kwargs):
        foreman_spec = dict(organization=dict(type='entity', required=True))
        foreman_spec.update(kwargs.pop('foreman_spec', {}))
        required_plugins = kwargs.pop('required_plugins', [])
        required_plugins.append(('katello', ['*']))
        super(KatelloMixin, self).__init__(foreman_spec=foreman_spec, required_plugins=required_plugins, **kwargs)