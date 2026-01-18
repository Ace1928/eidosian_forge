import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Tuple, Union
import grpc
from starlette.types import Receive, Scope, Send
from ray.actor import ActorHandle
from ray.serve._private.common import StreamingHTTPRequest, gRPCRequest
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import DEFAULT
from ray.serve.grpc_util import RayServegRPCContext
class ASGIProxyRequest(ProxyRequest):
    """ProxyRequest implementation to wrap ASGI scope, receive, and send."""

    def __init__(self, scope: Scope, receive: Receive, send: Send):
        self.scope = scope
        self.receive = receive
        self.send = send

    @property
    def request_type(self) -> str:
        return self.scope.get('type', '')

    @property
    def method(self) -> str:
        return self.scope.get('method', 'websocket').upper()

    @property
    def route_path(self) -> str:
        return self.scope.get('path', '')[len(self.root_path):]

    @property
    def is_route_request(self) -> bool:
        return self.route_path == '/-/routes'

    @property
    def is_health_request(self) -> bool:
        return self.route_path == '/-/healthz'

    @property
    def client(self) -> str:
        return self.scope.get('client', '')

    @property
    def root_path(self) -> str:
        return self.scope.get('root_path', '')

    @property
    def path(self) -> str:
        return self.scope.get('path', '')

    @property
    def headers(self) -> List[Tuple[bytes, bytes]]:
        return self.scope.get('headers', [])

    def set_path(self, path: str):
        self.scope['path'] = path

    def set_root_path(self, root_path: str):
        self.scope['root_path'] = root_path

    def request_object(self, proxy_handle) -> StreamingHTTPRequest:
        return StreamingHTTPRequest(pickled_asgi_scope=pickle.dumps(self.scope), http_proxy_handle=proxy_handle)