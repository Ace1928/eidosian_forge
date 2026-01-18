import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
@dataclasses.dataclass(frozen=True)
class CircuitWithCalibration:
    """Circuit with characterization data annotations.

    Attributes:
        circuit: Circuit instance.
        moment_to_calibration: Maps each moment within a circuit to an index of a characterization
            request or response. None means that there is no characterization data for that moment.
    """
    circuit: cirq.Circuit
    moment_to_calibration: Sequence[Optional[int]]