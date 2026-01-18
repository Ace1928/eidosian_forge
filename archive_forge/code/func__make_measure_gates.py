from qcs_api_client.models import InstructionSetArchitecture, Characteristic, Operation
from pyquil.external.rpcq import CompilerISA, add_edge, add_qubit, get_qubit, get_edge
import numpy as np
from pyquil.external.rpcq import (
from typing import List, Union, cast, DefaultDict, Set, Optional
from collections import defaultdict
def _make_measure_gates(node_id: int, characteristics: List[Characteristic]) -> List[MeasureInfo]:
    duration = _operation_names_to_compiler_duration_default[Supported1QGate.MEASURE]
    fidelity = _operation_names_to_compiler_fidelity_default[Supported1QGate.MEASURE]
    for characteristic in characteristics:
        if characteristic.name == 'fRO':
            fidelity = characteristic.value
            break
    return [MeasureInfo(operator=Supported1QGate.MEASURE, qubit=str(node_id), target='_', fidelity=fidelity, duration=duration), MeasureInfo(operator=Supported1QGate.MEASURE, qubit=str(node_id), target=None, fidelity=fidelity, duration=duration)]