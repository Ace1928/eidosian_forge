import sys
import logging
import queue
import threading
import time
import grpc
from typing import TYPE_CHECKING
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.debug import log_once
def _log_main(self) -> None:
    reconnecting = False
    while not self.client_worker._in_shutdown:
        if reconnecting:
            self.request_queue = queue.Queue()
            if self.last_req:
                self.request_queue.put(self.last_req)
        stub = ray_client_pb2_grpc.RayletLogStreamerStub(self.client_worker.channel)
        try:
            log_stream = stub.Logstream(iter(self.request_queue.get, None), metadata=self._metadata)
        except ValueError:
            time.sleep(0.5)
            continue
        try:
            for record in log_stream:
                if record.level < 0:
                    self.stdstream(level=record.level, msg=record.msg)
                self.log(level=record.level, msg=record.msg)
            return
        except grpc.RpcError as e:
            reconnecting = self._process_rpc_error(e)
            if not reconnecting:
                return