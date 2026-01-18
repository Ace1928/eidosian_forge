from __future__ import (absolute_import, division, print_function)
import json
import os
import socket
import uuid
import re
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url, HAS_GSSAPI
from ansible.module_utils.basic import env_fallback, AnsibleFallbackNotFound
def _env_then_dns_fallback(*args, **kwargs):
    """ Load value from environment or DNS in that order"""
    try:
        result = env_fallback(*args, **kwargs)
        if result == '':
            raise AnsibleFallbackNotFound
        return result
    except AnsibleFallbackNotFound:
        try:
            return socket.gethostbyaddr(socket.gethostbyname('ipa-ca'))[0]
        except Exception:
            raise AnsibleFallbackNotFound