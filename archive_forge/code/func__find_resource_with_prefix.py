import os
import hashlib
from typing import Any, Dict, List, Optional
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
def _find_resource_with_prefix(self, prefix: str, kind: str, api_version: str) -> Resource:
    for attribute in ['kind', 'name', 'singular_name']:
        try:
            return self.client.resources.get(**{'prefix': prefix, 'api_version': api_version, attribute: kind})
        except (ResourceNotFoundError, ResourceNotUniqueError):
            pass
    return self.client.resources.get(prefix=prefix, api_version=api_version, short_names=[kind])