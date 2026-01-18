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
class SockServerInterfaceReaderThread(threading.Thread):
    _socket_client: SockClient
    _stopped: 'Event'

    def __init__(self, clients: ClientDict, iface: 'InterfaceRelay', stopped: 'Event') -> None:
        self._iface = iface
        self._clients = clients
        threading.Thread.__init__(self)
        self.name = 'SockSrvIntRdThr'
        self._stopped = stopped

    def run(self) -> None:
        assert self._iface.relay_q
        while not self._stopped.is_set():
            try:
                result = self._iface.relay_q.get(timeout=1)
            except queue.Empty:
                continue
            except OSError:
                break
            except ValueError:
                break
            tracelog.log_message_dequeue(result, self._iface.relay_q)
            sockid = result.control.relay_id
            assert sockid
            sock_client = self._clients.get_client(sockid)
            assert sock_client
            sresp = spb.ServerResponse()
            sresp.result_communicate.CopyFrom(result)
            sock_client.send_server_response(sresp)