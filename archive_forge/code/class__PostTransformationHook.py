from typing import Dict, cast, Optional, Tuple, List, Callable
from pyquil import Program
import cirq
from cirq_rigetti.quil_output import RigettiQCSQuilOutput
from typing_extensions import Protocol
class _PostTransformationHook(Protocol):

    def __call__(self, *, program: Program, measurement_id_map: Dict[str, str]) -> Tuple[Program, Dict[str, str]]:
        pass