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
def _build_service_results(mock_qpu_implementer: Any, circuit: cirq.Circuit, sweepable: cirq.Sweepable, *, executor: executors.CircuitSweepExecutor=_default_executor, transformer: transformers.CircuitTransformer=transformers.default) -> Tuple[Sequence[cirq.Result], QuantumComputer, List[np.ndarray], List[cirq.ParamResolver]]:
    repetitions = 2
    param_resolvers = [r for r in cirq.to_resolvers(sweepable)]
    param_resolver_index = min(1, len(param_resolvers) - 1)
    param_resolver = param_resolvers[param_resolver_index]
    expected_results = [np.ones((repetitions,)) * (param_resolver['t'] if 't' in param_resolver else param_resolver_index)]
    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(expected_results)
    service = RigettiQCSService(quantum_computer=quantum_computer, executor=executor, transformer=transformer)
    result = service.run(circuit=circuit, param_resolver=param_resolver, repetitions=repetitions)
    return ([result], quantum_computer, expected_results, [param_resolver])