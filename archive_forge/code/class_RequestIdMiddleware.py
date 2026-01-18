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
class RequestIdMiddleware:

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        headers = MutableHeaders(scope=scope)
        if 'x-request-id' not in headers:
            request_id = generate_request_id()
            headers.append('x-request-id', request_id)
        elif 'x-request-id' in headers:
            request_id = headers['x-request-id']

        async def send_with_request_id(message: Dict):
            if message['type'] == 'http.response.start':
                headers = MutableHeaders(scope=message)
                headers.append('X-Request-ID', request_id)
            if message['type'] == 'websocket.accept':
                message['X-Request-ID'] = request_id
            await send(message)
        await self.app(scope, receive, send_with_request_id)