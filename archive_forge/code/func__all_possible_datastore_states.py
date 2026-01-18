import itertools
from collections import defaultdict
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, ops, protocols, value
from cirq.linalg import transformations
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.synchronize_terminal_measurements import find_terminal_measurements
def _all_possible_datastore_states(keys: Iterable[Tuple['cirq.MeasurementKey', int]], measurement_qubits: Dict['cirq.MeasurementKey', List[Tuple['cirq.Qid', ...]]]) -> Iterable['cirq.ClassicalDataStoreReader']:
    """The cartesian product of all possible DataStore states for the given keys."""
    all_possible_measurements = itertools.product(*[tuple(itertools.product(*[range(q.dimension) for q in measurement_qubits[k][i]])) for k, i in keys])
    for measurement_list in all_possible_measurements:
        records = {key: [(0,) * len(qubits) for qubits in qubits_list] for key, qubits_list in measurement_qubits.items()}
        for (k, i), measurement in zip(keys, measurement_list):
            records[k][i] = measurement
        yield value.ClassicalDataDictionaryStore(_records=records, _measured_qubits=measurement_qubits)