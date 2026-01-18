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
class KatelloScopedMixin(KatelloMixin):
    """
    Enhances :class:`KatelloMixin` with scoping by ``organization`` as required by Katello.
    """

    def __init__(self, **kwargs):
        entity_opts = kwargs.pop('entity_opts', {})
        if 'scope' not in entity_opts:
            entity_opts['scope'] = ['organization']
        elif 'organization' not in entity_opts['scope']:
            entity_opts['scope'].append('organization')
        super(KatelloScopedMixin, self).__init__(entity_opts=entity_opts, **kwargs)