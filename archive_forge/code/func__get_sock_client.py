from typing import TYPE_CHECKING, Optional
from wandb.proto import wandb_server_pb2 as spb
from ..lib.sock_client import SockClient
from .service_base import ServiceInterface
def _get_sock_client(self) -> SockClient:
    return self._sock_client