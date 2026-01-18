from __future__ import (absolute_import, division, print_function)
import re
import time
import json
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.parsing.ajson import AnsibleJSONEncoder, AnsibleJSONDecoder
from ansible.plugins.cache import BaseCacheModule
from ansible.utils.display import Display
@staticmethod
def _parse_connection(re_patt, uri):
    match = re_patt.match(uri)
    if not match:
        raise AnsibleError('Unable to parse connection string')
    return match.groups()