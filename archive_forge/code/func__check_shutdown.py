import math
import logging
import queue
import threading
import warnings
import grpc
from collections import OrderedDict
from typing import Any, Callable, Dict, TYPE_CHECKING, Optional, Union
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.client.common import (
from ray.util.debug import log_once
def _check_shutdown(self):
    assert self.lock.locked()
    if not self._in_shutdown:
        return
    self.lock.release()
    if threading.current_thread().ident == self.data_thread.ident:
        return
    from ray.util import disconnect
    disconnect()
    self.lock.acquire()
    if self._last_exception is not None:
        msg = f"Request can't be sent because the Ray client has already been disconnected due to an error. Last exception: {self._last_exception}"
    else:
        msg = "Request can't be sent because the Ray client has already been disconnected."
    raise ConnectionError(msg)