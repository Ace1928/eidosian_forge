from typing import List, Optional, Sequence
import cirq
def _estimate_run_time_seconds(width: int, depth: int, sweeps: int, repetitions: int, latency: Optional[float]=_BASE_LATENCY) -> float:
    """Returns an approximate number of seconds for execution of a single circuit.

    This includes the total cost of set up (initial time), cost per repetition (rep_time),
    and a base end-to-end latency cost of operation (configurable).


    Args:
        width: number of qubits
        depth: number of moments
        sweeps: total number of parameter sweeps
        repetitions: number of repetitions per parameter sweep
        latency: Optional latency to add (defaults to 1.5 seconds)
    """
    init_time = _initial_time(width, depth, sweeps)
    rep_time = _rep_time(width, depth, sweeps, repetitions)
    return rep_time + init_time + latency