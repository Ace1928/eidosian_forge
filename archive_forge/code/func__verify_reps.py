from typing import Callable, Sequence, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.engine.validating_sampler import VALIDATOR_TYPE
from cirq_google.serialization.serializer import Serializer
from cirq_google.api import v2
def _verify_reps(sweeps: Sequence[cirq.Sweepable], repetitions: Union[int, Sequence[int]], max_repetitions: int=MAX_TOTAL_REPETITIONS) -> None:
    """Verify that the total number of repetitions is under the limit."""
    total_reps = 0
    for idx, sweep in enumerate(sweeps):
        if not isinstance(repetitions, int):
            total_reps += len(list(cirq.to_resolvers(sweep))) * repetitions[idx]
        else:
            total_reps += len(list(cirq.to_resolvers(sweep))) * repetitions
    if total_reps > max_repetitions:
        raise RuntimeError('No requested processors currently support the number of requested total repetitions.')