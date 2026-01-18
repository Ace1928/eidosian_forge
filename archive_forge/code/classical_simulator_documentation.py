from typing import Dict
from collections import defaultdict
from cirq.sim.simulator import SimulatesSamples
from cirq import ops, protocols
from cirq.study.resolver import ParamResolver
from cirq.circuits.circuit import AbstractCircuit
from cirq.ops.raw_types import Qid
import numpy as np
A simulator that accepts only gates with classical counterparts.

    This simulator evolves a single state, using only gates that output a single state for each
    input state. The simulator runs in linear time, at the cost of not supporting superposition.
    It can be used to estimate costs and simulate circuits for simple non-quantum algorithms using
    many more qubits than fully capable quantum simulators.

    The supported gates are:
        - cirq.X
        - cirq.CNOT
        - cirq.SWAP
        - cirq.TOFFOLI
        - cirq.measure

    Args:
        circuit: The circuit to simulate.
        param_resolver: Parameters to run with the program.
        repetitions: Number of times to repeat the run. It is expected that
            this is validated greater than zero before calling this method.

    Returns:
        A dictionary mapping measurement keys to measurement results.

    Raises:
        ValueError: If
            - one of the gates is not an X, CNOT, SWAP, TOFFOLI or a measurement.
            - A measurement key is used for measurements on different numbers of qubits.
    