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
class DebugThread(threading.Thread):

    def __init__(self, mux: 'StreamMux') -> None:
        threading.Thread.__init__(self)
        self.daemon = True
        self.name = 'DebugThr'

    def run(self) -> None:
        while True:
            time.sleep(30)
            for thread in threading.enumerate():
                print(f'DEBUG: {thread.name}')