import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _get_node_provider(provider_config: Dict[str, Any], cluster_name: str, use_cache: bool=True) -> Any:
    """Get the instantiated node provider for a given provider config.

    Note that this may be used by private node providers that proxy methods to
    built-in node providers, so we should maintain backwards compatibility.

    Args:
        provider_config: provider section of the autoscaler config.
        cluster_name: cluster name from the autoscaler config.
        use_cache: whether or not to use a cached definition if available. If
            False, the returned object will also not be stored in the cache.

    Returns:
        NodeProvider
    """
    provider_key = (json.dumps(provider_config, sort_keys=True), cluster_name)
    if use_cache and provider_key in _provider_instances:
        return _provider_instances[provider_key]
    provider_cls = _get_node_provider_cls(provider_config)
    new_provider = provider_cls(provider_config, cluster_name)
    if use_cache:
        _provider_instances[provider_key] = new_provider
    return new_provider