from typing import Dict, List, Optional, Tuple, Union
import ray
from ray.serve import context
from ray.util import metrics
from ray.util.annotations import PublicAPI
def _add_serve_context_tag_values(tag_keys: Tuple, tags: Dict[str, str]):
    """Add serve context tag values to the metric tags"""
    _request_context = ray.serve.context._serve_request_context.get()
    if ROUTE_TAG in tag_keys and ROUTE_TAG not in tags:
        tags[ROUTE_TAG] = _request_context.route