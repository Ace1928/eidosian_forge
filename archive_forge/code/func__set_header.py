import os
import hashlib
from typing import Any, Dict, List, Optional
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
def _set_header(client, header, value):
    if isinstance(value, list):
        for v in value:
            client.set_default_header(header_name=unique_string(header), header_value=v)
    else:
        client.set_default_header(header_name=header, header_value=value)