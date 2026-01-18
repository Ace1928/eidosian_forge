from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def FENCE(*qubits: Union[int, Qubit, FormalArgument]) -> Union[FenceAll, Fence]:
    """
    Produce a FENCE instruction.

    Note: If no qubits are specified, then this is interpreted as a global FENCE.

    :params qubits: A list of qubits or formal arguments.
    :returns: A Fence or FenceAll instance.
    """
    if qubits:
        return Fence([Qubit(t) if isinstance(t, int) else t for t in qubits])
    else:
        return FenceAll()