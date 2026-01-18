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
def _data_main(self) -> None:
    reconnecting = False
    try:
        while not self.client_worker._in_shutdown:
            stub = ray_client_pb2_grpc.RayletDataStreamerStub(self.client_worker.channel)
            metadata = self._metadata + [('reconnecting', str(reconnecting))]
            resp_stream = stub.Datapath(self._requests(), metadata=metadata, wait_for_ready=True)
            try:
                for response in resp_stream:
                    self._process_response(response)
                return
            except grpc.RpcError as e:
                reconnecting = self._can_reconnect(e)
                if not reconnecting:
                    self._last_exception = e
                    return
                self._reconnect_channel()
    except Exception as e:
        self._last_exception = e
    finally:
        logger.debug('Shutting down data channel.')
        self._shutdown()