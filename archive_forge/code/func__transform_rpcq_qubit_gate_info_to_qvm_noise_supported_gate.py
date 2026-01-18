from pyquil.quilbase import Gate
from pyquil.quilatom import Parameter, unpack_qubit
from pyquil.external.rpcq import CompilerISA, Supported1QGate, Supported2QGate, GateInfo, Edge
from typing import List, Optional
import logging
def _transform_rpcq_qubit_gate_info_to_qvm_noise_supported_gate(qubit_id: int, gate: GateInfo) -> Optional[Gate]:
    if gate.operator == Supported1QGate.RX:
        if len(gate.parameters) == 1 and gate.parameters[0] == 0.0:
            return None
        parameters = [Parameter(param) if isinstance(param, str) else param for param in gate.parameters]
        return Gate(gate.operator, parameters, [unpack_qubit(qubit_id)])
    if gate.operator == Supported1QGate.RZ:
        return Gate(Supported1QGate.RZ, [Parameter('theta')], [unpack_qubit(qubit_id)])
    if gate.operator == Supported1QGate.I:
        return Gate(Supported1QGate.I, [], [unpack_qubit(qubit_id)])
    _log.warning('Unknown qubit gate operator: {}'.format(gate.operator))
    return None