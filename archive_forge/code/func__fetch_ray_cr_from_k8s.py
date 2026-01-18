import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
def _fetch_ray_cr_from_k8s(self) -> Dict[str, Any]:
    result = requests.get(self._ray_cr_url, headers=self._headers, verify=self._verify)
    if not result.status_code == 200:
        result.raise_for_status()
    ray_cr = result.json()
    return ray_cr