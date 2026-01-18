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
def _shutdown(self) -> None:
    """
        Shutdown the data channel
        """
    with self.lock:
        self._in_shutdown = True
        self.cv.notify_all()
        callbacks = self.asyncio_waiting_data.values()
        self.asyncio_waiting_data = {}
    if self._last_exception:
        err = ConnectionError(f'Failed during this or a previous request. Exception that broke the connection: {self._last_exception}')
    else:
        err = ConnectionError('Request cannot be fulfilled because the data client has disconnected.')
    for callback in callbacks:
        if callback:
            callback(err)