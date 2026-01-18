import multiprocessing
from typing import Dict, Any, Optional
from typing import Sequence
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def _ref_simulate_2q_xeb_circuit(task: Dict[str, Any]):
    """Helper function for simulating a given (circuit, cycle_depth)."""
    circuit_i = task['circuit_i']
    cycle_depth = task['cycle_depth']
    circuit = task['circuit']
    param_resolver = task['param_resolver']
    circuit_depth = cycle_depth * 2 + 1
    assert circuit_depth <= len(circuit)
    tcircuit = circuit[:circuit_depth]
    tcircuit = cirq.resolve_parameters_once(tcircuit, param_resolver=param_resolver)
    pure_sim = cirq.Simulator()
    psi = pure_sim.simulate(tcircuit)
    psi_vector = psi.final_state_vector
    pure_probs = cirq.state_vector_to_probabilities(psi_vector)
    return {'circuit_i': circuit_i, 'cycle_depth': cycle_depth, 'pure_probs': pure_probs}