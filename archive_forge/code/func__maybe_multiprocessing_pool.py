import contextlib
import multiprocessing
import multiprocessing.pool
from typing import Optional, Union, Iterator
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
import cirq.experiments.xeb_fitting as xebf
import cirq.experiments.xeb_sampling as xebsamp
from cirq_google.calibration.phased_fsim import (
@contextlib.contextmanager
def _maybe_multiprocessing_pool(n_processes: Optional[int]=None) -> Iterator[Union['multiprocessing.pool.Pool', None]]:
    """Yield a multiprocessing.Pool as a context manager, unless n_processes=1; then yield None,
    which should disable multiprocessing in XEB apis."""
    if n_processes == 1:
        yield None
        return
    with multiprocessing.get_context('spawn').Pool(processes=n_processes) as pool:
        yield pool