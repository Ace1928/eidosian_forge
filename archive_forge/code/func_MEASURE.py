from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def MEASURE(qubit: QubitDesignator, classical_reg: Optional[MemoryReferenceDesignator]) -> Measurement:
    """
    Produce a MEASURE instruction.

    :param qubit: The qubit to measure.
    :param classical_reg: The classical register to measure into, or None.
    :return: A Measurement instance.
    """
    qubit = unpack_qubit(qubit)
    if classical_reg is None:
        address = None
    else:
        address = unpack_classical_reg(classical_reg)
    return Measurement(qubit, address)