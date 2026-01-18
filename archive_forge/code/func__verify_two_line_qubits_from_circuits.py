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
def _verify_two_line_qubits_from_circuits(circuits: Sequence['cirq.Circuit']):
    if _verify_and_get_two_qubits_from_circuits(circuits) != devices.LineQubit.range(2):
        raise ValueError('`circuits` should be a sequence of circuits each operating on LineQubit(0) and LineQubit(1)')