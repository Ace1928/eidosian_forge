from typing import Any, List, Sequence, Tuple
import cirq
import pytest
from pyquil import Program
from pyquil.api import QuantumComputer
import numpy as np
from pyquil.gates import MEASURE, RX, X, DECLARE, H, CNOT
from cirq_rigetti import RigettiQCSService
from typing_extensions import Protocol
from cirq_rigetti import circuit_transformers as transformers
from cirq_rigetti import circuit_sweep_executors as executors
def _build_sampler_results(mock_qpu_implementer: Any, circuit: cirq.Circuit, sweepable: cirq.Sweepable, *, executor: executors.CircuitSweepExecutor=_default_executor, transformer: transformers.CircuitTransformer=transformers.default) -> Tuple[Sequence[cirq.Result], QuantumComputer, List[np.ndarray], cirq.Sweepable]:
    repetitions = 2
    param_resolvers = [r for r in cirq.to_resolvers(sweepable)]
    expected_results = [np.ones((repetitions,)) * (params['t'] if 't' in params else i) for i, params in enumerate(param_resolvers)]
    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(expected_results)
    service = RigettiQCSService(quantum_computer=quantum_computer, executor=executor, transformer=transformer)
    sampler = service.sampler()
    results = sampler.run_sweep(program=circuit, params=param_resolvers, repetitions=repetitions)
    return (results, quantum_computer, expected_results, param_resolvers)