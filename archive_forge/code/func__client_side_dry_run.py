from typing import Any, Dict, List, Optional, Tuple
from ansible_collections.kubernetes.core.plugins.module_utils.hashes import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.waiter import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils.common.dict_transformations import dict_merge
@property
def _client_side_dry_run(self):
    return self.module.check_mode and (not self.client.dry_run)