from typing import Dict, List, Optional, Tuple, Union
import ray
from ray.serve import context
from ray.util import metrics
from ray.util.annotations import PublicAPI
def _add_serve_metric_tags(tag_keys: Optional[Tuple[str]]=None) -> Tuple[str]:
    """Add serve context tags to the tag_keys"""
    if tag_keys is None:
        tag_keys = tuple()
    if context._get_internal_replica_context() is None:
        return tag_keys
    if DEPLOYMENT_TAG in tag_keys:
        raise ValueError(f"'{DEPLOYMENT_TAG}' tag is reserved for Ray Serve metrics")
    if REPLICA_TAG in tag_keys:
        raise ValueError(f"'{REPLICA_TAG}' tag is reserved for Ray Serve metrics")
    if APPLICATION_TAG in tag_keys:
        raise ValueError(f"'{APPLICATION_TAG}' tag is reserved for Ray Serve metrics")
    ray_serve_tags = (DEPLOYMENT_TAG, REPLICA_TAG)
    if context._get_internal_replica_context().app_name:
        ray_serve_tags += (APPLICATION_TAG,)
    if tag_keys:
        tag_keys = ray_serve_tags + tag_keys
    else:
        tag_keys = ray_serve_tags
    return tag_keys