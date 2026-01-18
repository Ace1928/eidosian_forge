from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
@deprecated(version='3.5.1', reason='The correct instruction is SWAP-PHASES, not SWAP-PHASE')
def SWAP_PHASE(frameA: Frame, frameB: Frame) -> SwapPhases:
    """
    Alias of :func:`SWAP_PHASES`.
    """
    return SWAP_PHASES(frameA, frameB)