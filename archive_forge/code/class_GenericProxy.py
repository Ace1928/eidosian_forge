import asyncio
import json
import logging
import os
import pickle
import socket
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type
import grpc
import starlette
import starlette.routing
import uvicorn
from packaging import version
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware
from starlette.types import Receive
import ray
from ray import serve
from ray._private.utils import get_or_create_event_loop
from ray.actor import ActorHandle
from ray.serve._private.common import EndpointInfo, EndpointTag, NodeId, RequestProtocol
from ray.serve._private.constants import (
from ray.serve._private.grpc_util import DummyServicer, create_serve_grpc_server
from ray.serve._private.http_util import (
from ray.serve._private.logging_utils import (
from ray.serve._private.long_poll import LongPollClient, LongPollNamespace
from ray.serve._private.proxy_request_response import (
from ray.serve._private.proxy_response_generator import ProxyResponseGenerator
from ray.serve._private.proxy_router import (
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import call_function_from_import_path
from ray.serve.config import gRPCOptions
from ray.serve.generated.serve_pb2 import HealthzResponse, ListApplicationsResponse
from ray.serve.generated.serve_pb2_grpc import add_RayServeAPIServiceServicer_to_server
from ray.serve.handle import DeploymentHandle
from ray.serve.schema import LoggingConfig
from ray.util import metrics
class GenericProxy(ABC):
    """This class is served as the base class for different types of proxies.
    It contains all the common setup and methods required for running a proxy.

    The proxy subclass need to implement the following methods:
      - `protocol()`
      - `not_found()`
      - `draining_response()`
      - `routes_response()`
      - `health_response()`
      - `setup_request_context_and_handle()`
      - `send_request_to_replica()`
    """

    def __init__(self, controller_name: str, node_id: NodeId, node_ip_address: str, proxy_router_class: Type[ProxyRouter], request_timeout_s: Optional[float]=None, controller_actor: Optional[ActorHandle]=None, proxy_actor: Optional[ActorHandle]=None):
        self.request_timeout_s = request_timeout_s
        if self.request_timeout_s is not None and self.request_timeout_s < 0:
            self.request_timeout_s = None
        self._node_id = node_id
        ray.serve.context._set_internal_replica_context(app_name=None, deployment=None, replica_tag=None, servable_object=None, controller_name=controller_name)
        self.route_info: Dict[str, EndpointTag] = dict()
        self.self_actor_handle = proxy_actor or ray.get_runtime_context().current_actor
        self.asgi_receive_queues: Dict[str, ASGIMessageQueue] = dict()
        self.proxy_router = proxy_router_class(serve.get_deployment_handle, self.protocol)
        self.long_poll_client = LongPollClient(controller_actor or ray.get_actor(controller_name, namespace=SERVE_NAMESPACE), {LongPollNamespace.ROUTE_TABLE: self._update_routes}, call_in_event_loop=get_or_create_event_loop())
        self.request_counter = metrics.Counter(f'serve_num_{self.protocol.lower()}_requests', description=f'The number of {self.protocol} requests processed.', tag_keys=('route', 'method', 'application', 'status_code'))
        self.request_error_counter = metrics.Counter(f'serve_num_{self.protocol.lower()}_error_requests', description=f'The number of errored {self.protocol} responses.', tag_keys=('route', 'error_code', 'method', 'application'))
        self.deployment_request_error_counter = metrics.Counter(f'serve_num_deployment_{self.protocol.lower()}_error_requests', description=f'The number of errored {self.protocol} responses returned by each deployment.', tag_keys=('deployment', 'error_code', 'method', 'route', 'application'))
        self.processing_latency_tracker = metrics.Histogram(f'serve_{self.protocol.lower()}_request_latency_ms', description=f'The end-to-end latency of {self.protocol} requests (measured from the Serve {self.protocol} proxy).', boundaries=DEFAULT_LATENCY_BUCKET_MS, tag_keys=('method', 'route', 'application', 'status_code'))
        self.num_ongoing_requests_gauge = metrics.Gauge(name=f'serve_num_ongoing_{self.protocol.lower()}_requests', description=f'The number of ongoing requests in this {self.protocol} proxy.', tag_keys=('node_id', 'node_ip_address')).set_default_tags({'node_id': node_id, 'node_ip_address': node_ip_address})
        self._prevent_node_downscale_ref = ray.put('prevent_node_downscale_object')
        self._ongoing_requests = 0
        self._draining_start_time: Optional[float] = None
        getattr(ServeUsageTag, f'{self.protocol.upper()}_PROXY_USED').record('1')

    @property
    @abstractmethod
    def protocol(self) -> RequestProtocol:
        """Protocol used in the proxy.

        Each proxy needs to implement its own logic for setting up the protocol.
        """
        raise NotImplementedError

    def _is_draining(self) -> bool:
        """Whether is proxy actor is in the draining status or not."""
        return self._draining_start_time is not None

    def _update_routes(self, endpoints: Dict[EndpointTag, EndpointInfo]) -> None:
        self.route_info: Dict[str, EndpointTag] = dict()
        for endpoint, info in endpoints.items():
            route = info.route
            self.route_info[route] = endpoint
        self.proxy_router.update_routes(endpoints)

    def is_drained(self):
        """Check whether the proxy actor is drained or not.

        A proxy actor is drained if it has no ongoing requests
        AND it has been draining for more than
        `PROXY_MIN_DRAINING_PERIOD_S` seconds.
        """
        if not self._is_draining():
            return False
        return not self._ongoing_requests and time.time() - self._draining_start_time > PROXY_MIN_DRAINING_PERIOD_S

    def update_draining(self, draining: bool):
        """Update the draining status of the proxy.

        This is called by the proxy state manager
        to drain or un-drain the proxy actor.
        """
        if draining and (not self._is_draining()):
            logger.info(f'Start to drain the proxy actor on node {self._node_id}.', extra={'log_to_stderr': False})
            self._draining_start_time = time.time()
        if not draining and self._is_draining():
            logger.info(f'Stop draining the proxy actor on node {self._node_id}.', extra={'log_to_stderr': False})
            self._draining_start_time = None

    @abstractmethod
    async def not_found(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        raise NotImplementedError

    @abstractmethod
    async def draining_response(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        raise NotImplementedError

    @abstractmethod
    async def routes_response(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        raise NotImplementedError

    @abstractmethod
    async def health_response(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        raise NotImplementedError

    def _ongoing_requests_start(self):
        """Ongoing requests start.

        The current autoscale logic can downscale nodes with ongoing requests if the
        node doesn't have replicas and has no primary copies of objects in the object
        store. The counter and the dummy object reference will help to keep the node
        alive while draining requests, so they are not dropped unintentionally.
        """
        self._ongoing_requests += 1
        self.num_ongoing_requests_gauge.set(self._ongoing_requests)

    def _ongoing_requests_end(self):
        """Ongoing requests end.

        Decrement the ongoing request counter and drop the dummy object reference
        signaling that the node can be downscaled safely.
        """
        self._ongoing_requests -= 1
        self.num_ongoing_requests_gauge.set(self._ongoing_requests)

    def _get_response_handler_info(self, proxy_request: ProxyRequest) -> ResponseHandlerInfo:
        if proxy_request.is_route_request:
            if self._is_draining():
                return ResponseHandlerInfo(response_generator=self.draining_response(proxy_request), metadata=HandlerMetadata(), should_record_access_log=False, should_record_request_metrics=False, should_increment_ongoing_requests=False)
            else:
                return ResponseHandlerInfo(response_generator=self.routes_response(proxy_request), metadata=HandlerMetadata(application_name='', deployment_name='', route=proxy_request.route_path), should_record_access_log=False, should_record_request_metrics=True, should_increment_ongoing_requests=False)
        elif proxy_request.is_health_request:
            if self._is_draining():
                return ResponseHandlerInfo(response_generator=self.draining_response(proxy_request), metadata=HandlerMetadata(), should_record_access_log=False, should_record_request_metrics=False, should_increment_ongoing_requests=False)
            else:
                return ResponseHandlerInfo(response_generator=self.health_response(proxy_request), metadata=HandlerMetadata(application_name='', deployment_name='', route=proxy_request.route_path), should_record_access_log=False, should_record_request_metrics=True, should_increment_ongoing_requests=False)
        else:
            matched_route = None
            if self.protocol == RequestProtocol.HTTP:
                matched_route = self.proxy_router.match_route(proxy_request.route_path)
            elif self.protocol == RequestProtocol.GRPC:
                matched_route = self.proxy_router.get_handle_for_endpoint(proxy_request.route_path)
            if matched_route is None:
                return ResponseHandlerInfo(response_generator=self.not_found(proxy_request), metadata=HandlerMetadata(application_name='', deployment_name='', route=proxy_request.route_path), should_record_access_log=True, should_record_request_metrics=True, should_increment_ongoing_requests=False)
            else:
                route_prefix, handle, app_is_cross_language = matched_route
                route_path = proxy_request.route_path
                if route_prefix != '/' and self.protocol == RequestProtocol.HTTP:
                    assert not route_prefix.endswith('/')
                    proxy_request.set_path(route_path.replace(route_prefix, '', 1))
                    proxy_request.set_root_path(proxy_request.root_path + route_prefix)
                handle, request_id = self.setup_request_context_and_handle(app_name=handle.deployment_id.app, handle=handle, route_path=route_path, proxy_request=proxy_request)
                response_generator = self.send_request_to_replica(request_id=request_id, handle=handle, proxy_request=proxy_request, app_is_cross_language=app_is_cross_language)
                return ResponseHandlerInfo(response_generator=response_generator, metadata=HandlerMetadata(application_name=handle.deployment_id.app, deployment_name=handle.deployment_id.name, route=route_path), should_record_access_log=True, should_record_request_metrics=True, should_increment_ongoing_requests=True)

    async def proxy_request(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        """Wrapper for proxy request.

        This method is served as common entry point by the proxy. It handles the
        routing, including routes and health checks, ongoing request counter,
        and metrics.
        """
        assert proxy_request.request_type in {'http', 'websocket', 'grpc'}
        response_handler_info = self._get_response_handler_info(proxy_request)
        start_time = time.time()
        if response_handler_info.should_increment_ongoing_requests:
            self._ongoing_requests_start()
        try:
            status: Optional[ResponseStatus] = None
            async for message in response_handler_info.response_generator:
                if isinstance(message, ResponseStatus):
                    status = message
                yield message
            assert status is not None and isinstance(status, ResponseStatus)
        finally:
            if response_handler_info.should_increment_ongoing_requests:
                self._ongoing_requests_end()
        latency_ms = (time.time() - start_time) * 1000.0
        if response_handler_info.should_record_access_log:
            logger.info(access_log_msg(method=proxy_request.method, status=str(status.code), latency_ms=latency_ms), extra={'log_to_stderr': False, 'serve_access_log': True})
        if response_handler_info.should_record_request_metrics:
            self.request_counter.inc(tags={'route': response_handler_info.metadata.route, 'method': proxy_request.method, 'application': response_handler_info.metadata.application_name, 'status_code': str(status.code)})
            self.processing_latency_tracker.observe(latency_ms, tags={'method': proxy_request.method, 'route': response_handler_info.metadata.route, 'application': response_handler_info.metadata.application_name, 'status_code': str(status.code)})
            if status.is_error:
                self.request_error_counter.inc(tags={'route': response_handler_info.metadata.route, 'error_code': str(status.code), 'method': proxy_request.method, 'application': response_handler_info.metadata.application_name})
                self.deployment_request_error_counter.inc(tags={'deployment': response_handler_info.metadata.deployment_name, 'error_code': str(status.code), 'method': proxy_request.method, 'route': response_handler_info.metadata.route, 'application': response_handler_info.metadata.application_name})

    @abstractmethod
    def setup_request_context_and_handle(self, app_name: str, handle: DeploymentHandle, route_path: str, proxy_request: ProxyRequest) -> Tuple[DeploymentHandle, str]:
        """Setup the request context and handle for the request.

        Each proxy needs to implement its own logic for setting up the request context
        and handle.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_request_to_replica(self, request_id: str, handle: DeploymentHandle, proxy_request: ProxyRequest, app_is_cross_language: bool=False) -> ResponseGenerator:
        """Send the request to the replica and handle streaming response.

        Each proxy needs to implement its own logic for sending the request and
        handling the streaming response.
        """
        raise NotImplementedError