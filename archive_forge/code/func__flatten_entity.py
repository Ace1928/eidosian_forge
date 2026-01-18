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
def _flatten_entity(entity, foreman_spec):
    """Flatten entity according to spec"""
    result = {}
    if entity is None:
        entity = {}
    for key, value in entity.items():
        if key in foreman_spec and foreman_spec[key].get('ensure', True) and (value is not None):
            spec = foreman_spec[key]
            flat_name = spec.get('flat_name', key)
            property_type = spec.get('type', 'str')
            if property_type == 'entity':
                if value is not NoEntity:
                    result[flat_name] = value['id']
                else:
                    result[flat_name] = None
            elif property_type == 'entity_list':
                result[flat_name] = sorted((val['id'] for val in value))
            elif property_type == 'nested_list':
                result[flat_name] = [_flatten_entity(ent, foreman_spec[key]['foreman_spec']) for ent in value]
            else:
                result[flat_name] = value
    return result