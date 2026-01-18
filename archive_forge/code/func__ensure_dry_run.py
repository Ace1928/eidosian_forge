import os
import hashlib
from typing import Any, Dict, List, Optional
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
def _ensure_dry_run(self, params: Dict) -> Dict:
    if self.dry_run:
        params['dry_run'] = self.K8S_SERVER_DRY_RUN
    return params