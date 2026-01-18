from __future__ import (absolute_import, division, print_function)
import base64
import os
import json
from stat import S_IRUSR, S_IWUSR
from ansible import constants as C
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
@staticmethod
def _encode_token(username, password):
    token = '%s:%s' % (to_text(username, errors='surrogate_or_strict'), to_text(password, errors='surrogate_or_strict', nonstring='passthru') or '')
    b64_val = base64.b64encode(to_bytes(token, encoding='utf-8', errors='surrogate_or_strict'))
    return to_text(b64_val)