import concurrent
import os
import time
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
import numpy as np
import pandas as pd
import tqdm
from cirq import ops, devices, value, protocols
from cirq.circuits import Circuit, Moment
from cirq.experiments.random_quantum_circuit_generation import CircuitLibraryCombination
def _generate_sample_2q_xeb_tasks(zipped_circuits: List[_ZippedCircuit], cycle_depths: Sequence[int]) -> List[_Sample2qXEBTask]:
    """Helper function used in `sample_2q_xeb_circuits` to prepare circuits in sampling tasks."""
    tasks: List[_Sample2qXEBTask] = []
    for cycle_depth in cycle_depths:
        for zipped_circuit in zipped_circuits:
            circuit_depth = cycle_depth * 2 + 1
            assert circuit_depth <= len(zipped_circuit.wide_circuit)
            prepared_circuit = zipped_circuit.wide_circuit[:circuit_depth]
            prepared_circuit += Moment((ops.measure(*pair, key=str(pair_i)) for pair_i, pair in enumerate(zipped_circuit.pairs)))
            tasks.append(_Sample2qXEBTask(cycle_depth=cycle_depth, layer_i=zipped_circuit.layer_i, combination_i=zipped_circuit.combination_i, prepared_circuit=prepared_circuit, combination=zipped_circuit.combination))
    return tasks