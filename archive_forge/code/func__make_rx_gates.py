from qcs_api_client.models import InstructionSetArchitecture, Characteristic, Operation
from pyquil.external.rpcq import CompilerISA, add_edge, add_qubit, get_qubit, get_edge
import numpy as np
from pyquil.external.rpcq import (
from typing import List, Union, cast, DefaultDict, Set, Optional
from collections import defaultdict
def _make_rx_gates(node_id: int, benchmarks: List[Operation]) -> List[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported1QGate.RX]
    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported1QGate.RX]
    gates = [GateInfo(operator=Supported1QGate.RX, parameters=[0.0], arguments=[node_id], fidelity=PERFECT_FIDELITY, duration=default_duration)]
    fidelity = _get_frb_sim_1q(node_id, benchmarks)
    if fidelity is None:
        fidelity = default_fidelity
    for param in [np.pi, -np.pi, np.pi / 2, -np.pi / 2]:
        gates.append(GateInfo(operator=Supported1QGate.RX, parameters=[param], arguments=[node_id], fidelity=fidelity, duration=default_duration))
    return gates