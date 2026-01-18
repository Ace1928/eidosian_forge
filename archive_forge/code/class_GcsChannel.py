import logging
from typing import Optional
from ray._private import ray_constants
import ray._private.gcs_aio_client
from ray.core.generated.common_pb2 import ErrorType, JobConfig
from ray.core.generated.gcs_pb2 import (
class GcsChannel:

    def __init__(self, gcs_address: Optional[str]=None, aio: bool=False):
        self._gcs_address = gcs_address
        self._aio = aio

    @property
    def address(self):
        return self._gcs_address

    def connect(self):
        self._channel = create_gcs_channel(self._gcs_address, self._aio)

    def channel(self):
        return self._channel