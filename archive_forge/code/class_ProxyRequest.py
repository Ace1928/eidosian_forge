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
class ProxyRequest(ABC):
    """Base ProxyRequest class to use in the common interface among proxies"""

    @property
    @abstractmethod
    def request_type(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def method(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def route_path(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_route_request(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_health_request(self) -> bool:
        raise NotImplementedError