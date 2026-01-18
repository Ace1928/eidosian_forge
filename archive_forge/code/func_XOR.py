from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def XOR(classical_reg1: MemoryReferenceDesignator, classical_reg2: Union[MemoryReferenceDesignator, int]) -> ClassicalExclusiveOr:
    """
    Produce an exclusive OR instruction.

    :param classical_reg1: The first classical register, which gets modified.
    :param classical_reg2: The second classical register or immediate value.
    :return: A ClassicalExclusiveOr instance.
    """
    left, right = unpack_reg_val_pair(classical_reg1, classical_reg2)
    assert isinstance(right, (MemoryReference, int))
    return ClassicalExclusiveOr(left, right)