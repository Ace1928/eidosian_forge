from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def SET_PHASE(frame: Frame, phase: ParameterDesignator) -> SetPhase:
    """
    Produce a SET-PHASE instruction.

    :param frame: The frame on which to set the phase.
    :param phase: The new phase value, in radians.
    :returns: A SetPhase instance.
    """
    return SetPhase(frame, phase)