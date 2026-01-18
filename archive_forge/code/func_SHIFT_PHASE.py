from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def SHIFT_PHASE(frame: Frame, phase: ParameterDesignator) -> ShiftPhase:
    """
    Produce a SHIFT-PHASE instruction.

    :param frame: The frame on which to shift the phase.
    :param phase: The value, in radians, to add to the existing phase.
    :returns: A ShiftPhase instance.
    """
    return ShiftPhase(frame, phase)