from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union
import pennylane as qml
from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape
from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher
def _batch_postprocessing(results: ResultBatch, individual_fns: List[PostProcessingFn], slices: List[slice]) -> ResultBatch:
    """Broadcast individual post processing functions onto their respective tapes.

    Args:
        results (ResultBatch): The numeric outcome from executing a batch of :class:`~.QuantumTape`

    Keyword Args:
        individual_fns (List[Callable]): postprocessing functions converting a batch of results into a single result
           corresponding to only a single :class:`~.QuantumTape`.
        slices (List[slice]): the indices for the results that correspond to each individual post processing function.

    >>> results = (1.0, 2.0, 3.0, 4.0)
    >>> def postprocessing1(results):
    ...     return results[0] + results[1]
    >>> def postprocessing2(results):
    ...     return results[0]+0.5
    >>> def postprocessing3(results):
    ...     return results[0]*2
    >>> slices = [slice(0,2), slice(2,3), slice(3,4)]
    >>> individual_fns = [postprocessing1, postprocessing2, postprocessing3]
    >>> _batch_postprocessing(results, individual_fns, slices)
    (3.0, 3.5, 8.0)

    """
    return tuple((fn(results[sl]) for fn, sl in zip(individual_fns, slices)))