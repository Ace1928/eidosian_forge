import datetime
import sys
from typing import (
import warnings
import duet
import proto
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.protobuf import any_pb2, field_mask_pb2
from google.protobuf.timestamp_pb2 import Timestamp
from cirq._compat import cached_property
from cirq._compat import deprecated_parameter
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine import stream_manager
def _ids_from_calibration_name(calibration_name: str) -> Tuple[str, str, int]:
    parts = calibration_name.split('/')
    return (parts[1], parts[3], int(parts[5]))