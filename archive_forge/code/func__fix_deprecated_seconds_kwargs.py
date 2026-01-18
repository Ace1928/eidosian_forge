import datetime
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.cloud import quantum
from cirq_google.api import v2
from cirq_google.devices import grid_device
from cirq_google.engine import (
def _fix_deprecated_seconds_kwargs(kwargs):
    if 'earliest_timestamp_seconds' in kwargs:
        kwargs['earliest_timestamp'] = kwargs['earliest_timestamp_seconds']
        del kwargs['earliest_timestamp_seconds']
    if 'latest_timestamp_seconds' in kwargs:
        kwargs['latest_timestamp'] = kwargs['latest_timestamp_seconds']
        del kwargs['latest_timestamp_seconds']
    return kwargs