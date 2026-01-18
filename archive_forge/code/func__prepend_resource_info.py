from typing import Dict
from ansible.module_utils._text import to_native
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.resource import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.service import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.waiter import exists
from ansible_collections.kubernetes.core.plugins.module_utils.selector import (
def _prepend_resource_info(resource, msg):
    return '%s %s: %s' % (resource['kind'], resource['metadata']['name'], msg)