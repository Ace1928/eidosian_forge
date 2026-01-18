from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def SET_SCALE(frame: Frame, scale: ParameterDesignator) -> SetScale:
    """
    Produce a SET-SCALE instruction.

    :param frame: The frame on which to set the scale.
    :param scale: The scaling factor.
    :returns: A SetScale instance.
    """
    return SetScale(frame, scale)