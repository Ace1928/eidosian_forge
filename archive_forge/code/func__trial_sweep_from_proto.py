from typing import cast, Dict, Hashable, Iterable, List, Optional, Sequence
from collections import OrderedDict
import dataclasses
import numpy as np
import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import result_pb2
def _trial_sweep_from_proto(msg: result_pb2.SweepResult, measure_map: Optional[Dict[str, MeasureInfo]]=None) -> Sequence[cirq.Result]:
    """Converts a SweepResult proto into List of list of trial results.

    Args:
        msg: v2 Result message to convert.
        measure_map: A mapping of measurement keys to a measurement
            configuration containing qubit ordering. If no measurement config is
            provided, then all results will be returned in the order specified
            within the result.

    Returns:
        A list containing a list of trial results for the sweep.

    Raises:
        ValueError: If a qubit already exists in the measurement results.
    """
    trial_sweep: List[cirq.Result] = []
    for pr in msg.parameterized_results:
        records: Dict[str, np.ndarray] = {}
        for mr in pr.measurement_results:
            instances = max(mr.instances, 1)
            qubit_results: OrderedDict[cirq.GridQubit, np.ndarray] = OrderedDict()
            for qmr in mr.qubit_measurement_results:
                qubit = v2.grid_qubit_from_proto_id(qmr.qubit.id)
                if qubit in qubit_results:
                    raise ValueError(f'Qubit already exists: {qubit}.')
                qubit_results[qubit] = unpack_bits(qmr.results, msg.repetitions * instances)
            if measure_map:
                ordered_results = [qubit_results[qubit] for qubit in measure_map[mr.key].qubits]
            else:
                ordered_results = list(qubit_results.values())
            shape = (msg.repetitions, instances, len(qubit_results))
            records[mr.key] = np.array(ordered_results).transpose().reshape(shape)
        trial_sweep.append(cirq.ResultDict(params=cirq.ParamResolver(dict(pr.params.assignments)), records=records))
    return trial_sweep