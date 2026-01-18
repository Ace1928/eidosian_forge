from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
@dataclass
class RunningTaskInfo:
    inputs: RefBundle
    num_outputs: int
    bytes_outputs: int