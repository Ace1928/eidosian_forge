from typing import Dict, List, Optional, Tuple, Union
import ray
from ray.serve import context
from ray.util import metrics
from ray.util.annotations import PublicAPI
def _add_serve_metric_default_tags(default_tags: Dict[str, str]):
    """Add serve context tags and values to the default_tags"""
    if context._get_internal_replica_context() is None:
        return default_tags
    if DEPLOYMENT_TAG in default_tags:
        raise ValueError(f"'{DEPLOYMENT_TAG}' tag is reserved for Ray Serve metrics")
    if REPLICA_TAG in default_tags:
        raise ValueError(f"'{REPLICA_TAG}' tag is reserved for Ray Serve metrics")
    if APPLICATION_TAG in default_tags:
        raise ValueError(f"'{APPLICATION_TAG}' tag is reserved for Ray Serve metrics")
    replica_context = context._get_internal_replica_context()
    default_tags[DEPLOYMENT_TAG] = replica_context.deployment
    default_tags[REPLICA_TAG] = replica_context.replica_tag
    if replica_context.app_name:
        default_tags[APPLICATION_TAG] = replica_context.app_name
    return default_tags