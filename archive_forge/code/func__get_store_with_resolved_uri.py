import inspect
from functools import lru_cache
from mlflow.tracking.registry import StoreRegistry
@lru_cache(maxsize=100)
def _get_store_with_resolved_uri(self, resolved_store_uri, resolved_tracking_uri):
    """
        Retrieve the store associated with a resolved (non-None) store URI and an artifact URI.
        Caching is done on resolved URIs because the meaning of an unresolved (None) URI may change
        depending on external configuration, such as environment variables
        """
    builder = self.get_store_builder(resolved_store_uri)
    builder_param_names = set(inspect.signature(builder).parameters.keys())
    if 'store_uri' in builder_param_names and 'tracking_uri' in builder_param_names:
        return builder(store_uri=resolved_store_uri, tracking_uri=resolved_tracking_uri)
    else:
        return builder(store_uri=resolved_store_uri)