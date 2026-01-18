import datetime
from typing import Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Union
import duet
from google.protobuf import any_pb2
import cirq
from cirq_google.engine import abstract_program, engine_client, util
from cirq_google.cloud import quantum
from cirq_google.engine.result_type import ResultType
from cirq_google.api import v2
from cirq_google.engine import engine_job
from cirq_google.serialization import circuit_serializer
def _deserialize_program(code: any_pb2.Any, program_num: Optional[int]=None) -> cirq.Circuit:
    import cirq_google.engine.engine as engine_base
    code_type = code.type_url[len(engine_base.TYPE_PREFIX):]
    program = None
    if code_type == 'cirq.google.api.v1.Program' or code_type == 'cirq.api.google.v1.Program':
        raise ValueError('deserializing a v1 Program is not supported')
    elif code_type == 'cirq.google.api.v2.Program' or code_type == 'cirq.api.google.v2.Program':
        program = v2.program_pb2.Program.FromString(code.value)
    elif code_type == 'cirq.google.api.v2.BatchProgram':
        if program_num is None:
            raise ValueError('A program number must be specified when deserializing a Batch Program')
        batch = v2.batch_pb2.BatchProgram.FromString(code.value)
        if abs(program_num) >= len(batch.programs):
            raise ValueError(f'Only {len(batch.programs)} in the batch but index {program_num} was specified')
        program = batch.programs[program_num]
    if program is not None:
        return circuit_serializer.CIRCUIT_SERIALIZER.deserialize(program)
    raise ValueError(f'unsupported program type: {code_type}')