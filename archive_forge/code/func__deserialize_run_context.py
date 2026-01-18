import datetime
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import duet
from google.protobuf import any_pb2
import cirq
from cirq_google.engine import abstract_job, calibration, engine_client
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.cloud import quantum
from cirq_google.engine.result_type import ResultType
from cirq_google.engine.engine_result import EngineResult
from cirq_google.api import v1, v2
def _deserialize_run_context(run_context: any_pb2.Any) -> Tuple[int, List[cirq.Sweep]]:
    import cirq_google.engine.engine as engine_base
    run_context_type = run_context.type_url[len(engine_base.TYPE_PREFIX):]
    if run_context_type == 'cirq.google.api.v1.RunContext' or run_context_type == 'cirq.api.google.v1.RunContext':
        raise ValueError('deserializing a v1 RunContext is not supported')
    if run_context_type == 'cirq.google.api.v2.RunContext' or run_context_type == 'cirq.api.google.v2.RunContext':
        v2_run_context = v2.run_context_pb2.RunContext.FromString(run_context.value)
        return (v2_run_context.parameter_sweeps[0].repetitions, [v2.sweep_from_proto(s.sweep) for s in v2_run_context.parameter_sweeps])
    raise ValueError(f'unsupported run_context type: {run_context_type}')