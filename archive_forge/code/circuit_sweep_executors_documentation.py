from typing import Any, cast, Dict, Optional, Sequence, Union
from pyquil import Program
from pyquil.api import QuantumComputer, QuantumExecutable
from pyquil.quilbase import Declare
import cirq
import sympy
from typing_extensions import Protocol
from cirq_rigetti.logging import logger
from cirq_rigetti import circuit_transformers as transformers
Transforms `cirq.Circuit` to `pyquil.Program` and executes it for given arguments.

        Args:
            quantum_computer: The `pyquil.api.QuantumComputer` against which to execute the circuit.
            circuit: The `cirq.Circuit` to transform into a `pyquil.Program` and executed on the
                `quantum_computer`.
            resolvers: A sequence of parameter resolvers that the executor must resolve.
            repetitions: Number of times to run each iteration through the `resolvers`. For a given
                resolver, the `cirq.Result` will include a measurement for each repetition.
            transformer: A callable that transforms the `cirq.Circuit` into a `pyquil.Program`.
                You may pass your own callable or any function from
                `cirq_rigetti.circuit_transformers`.

        Returns:
            A list of `cirq.Result`, each corresponding to a resolver in `resolvers`.
        