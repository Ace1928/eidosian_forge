import queue
import socket
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from wandb.proto import wandb_server_pb2 as spb
from wandb.sdk.internal.settings_static import SettingsStatic
from ..lib import tracelog
from ..lib.sock_client import SockClient, SockClientClosedError
from .streams import StreamMux
class ClientDict:
    _client_dict: Dict[str, SockClient]
    _lock: threading.Lock

    def __init__(self) -> None:
        self._client_dict = {}
        self._lock = threading.Lock()

    def get_client(self, client_id: str) -> Optional[SockClient]:
        with self._lock:
            client = self._client_dict.get(client_id)
        return client

    def add_client(self, client: SockClient) -> None:
        with self._lock:
            self._client_dict[client._sockid] = client

    def del_client(self, client: SockClient) -> None:
        with self._lock:
            del self._client_dict[client._sockid]