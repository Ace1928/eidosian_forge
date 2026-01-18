import datetime
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.cloud import quantum
from cirq_google.api import v2
from cirq_google.devices import grid_device
from cirq_google.engine import (
def _to_date_time_filters(from_time: Union[None, datetime.datetime, datetime.timedelta], to_time: Union[None, datetime.datetime, datetime.timedelta]) -> List[str]:
    now = datetime.datetime.now()
    if from_time is None:
        start_time = None
    elif isinstance(from_time, datetime.timedelta):
        start_time = now + from_time
    elif isinstance(from_time, datetime.datetime):
        start_time = from_time
    else:
        raise ValueError(f"Don't understand from_time of type {type(from_time)}.")
    if to_time is None:
        end_time = None
    elif isinstance(to_time, datetime.timedelta):
        end_time = now + to_time
    elif isinstance(to_time, datetime.datetime):
        end_time = to_time
    else:
        raise ValueError(f"Don't understand to_time of type {type(to_time)}.")
    filters = []
    if end_time is not None:
        filters.append(f'start_time < {int(end_time.timestamp())}')
    if start_time is not None:
        filters.append(f'end_time > {int(start_time.timestamp())}')
    return filters