from qcs_api_client.models import InstructionSetArchitecture, Characteristic, Operation
from pyquil.external.rpcq import CompilerISA, add_edge, add_qubit, get_qubit, get_edge
import numpy as np
from pyquil.external.rpcq import (
from typing import List, Union, cast, DefaultDict, Set, Optional
from collections import defaultdict
def _transform_qubit_operation_to_gates(operation_name: str, node_id: int, characteristics: List[Characteristic], benchmarks: List[Operation]) -> List[Union[GateInfo, MeasureInfo]]:
    if operation_name == Supported1QGate.RX:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_rx_gates(node_id, benchmarks))
    elif operation_name == Supported1QGate.RZ:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_rz_gates(node_id))
    elif operation_name == Supported1QGate.MEASURE:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_measure_gates(node_id, characteristics))
    elif operation_name == Supported1QGate.WILDCARD:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_wildcard_1q_gates(node_id))
    elif operation_name in {'I', 'RESET'}:
        return []
    else:
        raise QCSISAParseError('Unsupported qubit operation: {}'.format(operation_name))