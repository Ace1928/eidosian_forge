import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
def _generate_legacy_autoscaling_config_fields() -> Dict[str, Any]:
    """Generates legacy autoscaling config fields required for compatibiliy."""
    return {'file_mounts': {}, 'cluster_synced_files': [], 'file_mounts_sync_continuously': False, 'initialization_commands': [], 'setup_commands': [], 'head_setup_commands': [], 'worker_setup_commands': [], 'head_start_ray_commands': [], 'worker_start_ray_commands': [], 'auth': {}}