from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def CONVERT(classical_reg1: MemoryReferenceDesignator, classical_reg2: MemoryReferenceDesignator) -> ClassicalConvert:
    """
    Produce a CONVERT instruction.

    :param classical_reg1: MemoryReference to store to.
    :param classical_reg2: MemoryReference to read from.
    :return: A ClassicalConvert instance.
    """
    return ClassicalConvert(unpack_classical_reg(classical_reg1), unpack_classical_reg(classical_reg2))