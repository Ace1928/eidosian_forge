from pyquil.quilbase import Gate
from pyquil.quilatom import Parameter, unpack_qubit
from pyquil.external.rpcq import CompilerISA, Supported1QGate, Supported2QGate, GateInfo, Edge
from typing import List, Optional
import logging
def _transform_rpcq_edge_gate_info_to_qvm_noise_supported_gates(edge: Edge) -> List[Gate]:
    operators = [gate.operator for gate in edge.gates]
    targets = [unpack_qubit(t) for t in edge.ids]
    gates = []
    if Supported2QGate.CZ in operators:
        gates.append(Gate('CZ', [], targets))
        gates.append(Gate('CZ', [], targets[::-1]))
        return gates
    if Supported2QGate.ISWAP in operators:
        gates.append(Gate('ISWAP', [], targets))
        gates.append(Gate('ISWAP', [], targets[::-1]))
        return gates
    if Supported2QGate.CPHASE in operators:
        gates.append(Gate('CPHASE', [THETA], targets))
        gates.append(Gate('CPHASE', [THETA], targets[::-1]))
        return gates
    if Supported2QGate.XY in operators:
        gates.append(Gate('XY', [THETA], targets))
        gates.append(Gate('XY', [THETA], targets[::-1]))
        return gates
    if Supported2QGate.WILDCARD in operators:
        gates.append(Gate('_', '_', targets))
        gates.append(Gate('_', '_', targets[::-1]))
        return gates
    _log.warning(f'no gate for edge {edge.ids}')
    return gates