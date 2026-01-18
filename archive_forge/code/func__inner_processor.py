import datetime
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.cloud import quantum
from cirq_google.api import v2
from cirq_google.devices import grid_device
from cirq_google.engine import (
def _inner_processor(self) -> quantum.QuantumProcessor:
    if self._processor is None:
        self._processor = self.context.client.get_processor(self.project_id, self.processor_id)
    return self._processor