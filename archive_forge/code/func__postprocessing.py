from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.options import Options
from qiskit.result import QuasiDistribution, Result
from qiskit.transpiler.passmanager import PassManager
from .backend_estimator import _prepare_counts, _run_circuits
from .base import BaseSampler, SamplerResult
from .primitive_job import PrimitiveJob
from .utils import _circuit_key
def _postprocessing(self, result: list[Result], circuits: list[QuantumCircuit]) -> SamplerResult:
    counts = _prepare_counts(result)
    shots = sum(counts[0].values())
    probabilities = []
    metadata: list[dict[str, Any]] = [{} for _ in range(len(circuits))]
    for count in counts:
        prob_dist = {k: v / shots for k, v in count.items()}
        probabilities.append(QuasiDistribution(prob_dist, shots=shots, stddev_upper_bound=math.sqrt(1 / shots)))
        for metadatum in metadata:
            metadatum['shots'] = shots
    return SamplerResult(probabilities, metadata)