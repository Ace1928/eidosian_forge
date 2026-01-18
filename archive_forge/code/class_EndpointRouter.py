import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve.handle import RayServeHandle
class EndpointRouter(ProxyRouter):
    """Router that matches endpoint to return the handle."""

    def __init__(self, get_handle: Callable, protocol: RequestProtocol):
        self._get_handle = get_handle
        self._protocol = protocol
        self.handles: Dict[EndpointTag, RayServeHandle] = dict()
        self.endpoints: Dict[EndpointTag, EndpointInfo] = dict()

    def update_routes(self, endpoints: Dict[EndpointTag, EndpointInfo]):
        logger.info(f'Got updated endpoints: {endpoints}.', extra={'log_to_stderr': False})
        self.endpoints = endpoints
        existing_handles = set(self.handles.keys())
        for endpoint, info in endpoints.items():
            if endpoint in self.handles:
                existing_handles.remove(endpoint)
            else:
                handle = self._get_handle(endpoint.name, endpoint.app).options(stream=not info.app_is_cross_language, use_new_handle_api=True, _prefer_local_routing=RAY_SERVE_PROXY_PREFER_LOCAL_NODE_ROUTING)
                handle._set_request_protocol(self._protocol)
                self.handles[endpoint] = handle
        if len(existing_handles) > 0:
            logger.info(f'Deleting {len(existing_handles)} unused handles.', extra={'log_to_stderr': False})
        for endpoint in existing_handles:
            del self.handles[endpoint]

    def get_handle_for_endpoint(self, target_app_name: str) -> Optional[Tuple[str, RayServeHandle, bool]]:
        """Return the handle that matches with endpoint.

        Args:
            target_app_name: app_name to match against.
        Returns:
            (route, handle, app_name, is_cross_language) for the single app if there
            is only one, else find the app and handle for exact match. Else return None.
        """
        for endpoint_tag, handle in self.handles.items():
            if target_app_name == endpoint_tag.app or len(self.handles) == 1:
                endpoint_info = self.endpoints[endpoint_tag]
                return (endpoint_info.route, handle, endpoint_info.app_is_cross_language)
        return None