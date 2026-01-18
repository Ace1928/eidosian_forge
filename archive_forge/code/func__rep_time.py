from typing import List, Optional, Sequence
import cirq
def _rep_time(width: int, depth: int, sweeps: int, reps: int) -> float:
    """Estimated time of executing repetitions.

    This includes all incremental costs of executing a repetition and of
    sending data back and forth from the electronics.

    This is based on an approximate rep rate for "fast" circuits at about
    24k reps per second.  More qubits measured (width) primarily slows
    this down, with an additional factor for very high depth circuits.

    For multiple sweeps, there is some additional cost, since not all
    sweeps can be batched together.  Sweeping in general is more efficient,
    but it is not perfectly parallel.  Sweeping also seems to be more
    sensitive to the number of qubits measured, for reasons not understood.

    Args:
        width: number of qubits
        depth: number of moments
        sweeps: total number of parameter sweeps
        reps: number of repetitions per parameter sweep
    """
    total_reps = sweeps * reps
    rep_rate = 24000 / (0.9 + width / 38) / (0.9 + depth / 5000)
    if sweeps > 1:
        rep_rate *= 0.72
        if width < 55:
            rep_rate *= 1 - (width - 25) / 40
        else:
            rep_rate *= 0.25
    return total_reps / rep_rate