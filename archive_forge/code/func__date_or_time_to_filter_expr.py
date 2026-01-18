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
def _date_or_time_to_filter_expr(param_name: str, param: Union[datetime.datetime, datetime.date]):
    """Formats datetime or date to filter expressions.

    Args:
        param_name: The name of the filter parameter (for error messaging).
        param: The value of the parameter.

    Raises:
        ValueError: If the supplied param is not a datetime or date.
    """
    if isinstance(param, datetime.datetime):
        return f'{int(param.timestamp())}'
    elif isinstance(param, datetime.date):
        return f'{param.isoformat()}'
    raise ValueError(f'Unsupported date/time type for {param_name}: got {param} of type {type(param)}. Supported types: datetime.datetime anddatetime.date')