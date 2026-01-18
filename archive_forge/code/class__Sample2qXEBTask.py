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
@dataclass(frozen=True)
class _Sample2qXEBTask:
    """Helper container for grouping a circuit to be sampled.

    `prepared_circuit` is the full-length circuit (with index `circuit_i`) that has been truncated
    to `cycle_depth` and has a measurement gate on it.
    """
    cycle_depth: int
    layer_i: int
    combination_i: int
    prepared_circuit: 'cirq.AbstractCircuit'
    combination: List[int]