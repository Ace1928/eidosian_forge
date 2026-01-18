import dataclasses
from typing import cast, Optional, Sequence, Union
import cirq
from cirq_google import ProcessorSampler, get_engine
from cirq_google.engine import (
@dataclasses.dataclass
class QCSObjectsForNotebook:
    """All the objects you might need to run a notbook with QCS.

    Contains an (Abstract) Engine, Processor, Device, and Sampler,
    as well as associated meta-data signed_in, processor_id, and project_id.

    This removes the need for boiler plate in notebooks, and provides a
    central place to handle the various environments (testing vs production),
    (stand-alone vs colab vs jupyter).
    """
    engine: AbstractEngine
    processor: AbstractProcessor
    device: cirq.Device
    sampler: ProcessorSampler
    signed_in: bool
    processor_id: Optional[str]
    project_id: Optional[str]
    is_simulator: bool