import os
from typing import Generator, Callable, Union, Sequence, Optional
from copy import copy
import warnings
import pennylane as qml
from pennylane import Snapshot
from pennylane.operation import Tensor, StatePrepBase
from pennylane.measurements import (
from pennylane.typing import ResultBatch, Result
from pennylane import DeviceError
from pennylane import transform
from pennylane.wires import WireError
def _operator_decomposition_gen(op: qml.operation.Operator, acceptance_function: Callable[[qml.operation.Operator], bool], decomposer: Callable[[qml.operation.Operator], Sequence[qml.operation.Operator]], max_expansion: Optional[int]=None, current_depth=0, name: str='device') -> Generator[qml.operation.Operator, None, None]:
    """A generator that yields the next operation that is accepted."""
    max_depth_reached = False
    if max_expansion is not None and max_expansion <= current_depth:
        max_depth_reached = True
    if acceptance_function(op) or max_depth_reached:
        yield op
    else:
        try:
            decomp = decomposer(op)
            current_depth += 1
        except qml.operation.DecompositionUndefinedError as e:
            raise DeviceError(f'Operator {op} not supported on {name} and does not provide a decomposition.') from e
        for sub_op in decomp:
            yield from _operator_decomposition_gen(sub_op, acceptance_function, decomposer=decomposer, max_expansion=max_expansion, current_depth=current_depth, name=name)