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
def _calibration_name_from_ids(project_id: str, processor_id: str, calibration_time_seconds: int) -> str:
    return f'projects/{project_id}/processors/{processor_id}/calibrations/{calibration_time_seconds}'