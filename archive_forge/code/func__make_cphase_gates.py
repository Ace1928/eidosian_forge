from qcs_api_client.models import InstructionSetArchitecture, Characteristic, Operation
from pyquil.external.rpcq import CompilerISA, add_edge, add_qubit, get_qubit, get_edge
import numpy as np
from pyquil.external.rpcq import (
from typing import List, Union, cast, DefaultDict, Set, Optional
from collections import defaultdict
def _make_cphase_gates(characteristics: List[Characteristic]) -> List[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported2QGate.CPHASE]
    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported2QGate.CPHASE]
    fidelity = default_fidelity
    for characteristic in characteristics:
        if characteristic.name == 'fCPHASE':
            fidelity = characteristic.value
            break
    return [GateInfo(operator=Supported2QGate.CPHASE, parameters=['theta'], arguments=['_', '_'], fidelity=fidelity, duration=default_duration)]