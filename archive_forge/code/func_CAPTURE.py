from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def CAPTURE(frame: Frame, kernel: Waveform, memory_region: MemoryReferenceDesignator, nonblocking: bool=False) -> Capture:
    """
    Produce a CAPTURE instruction.

    :param frame: The frame on which to capture an IQ value.
    :param kernel: The integrating kernel for the capture.
    :param memory_region: The classical memory region to store the resulting IQ value.
    :param nonblocking: A flag indicating whether the capture is NONBLOCKING.
    :returns: A Capture instance.
    """
    memory_region = unpack_classical_reg(memory_region)
    return Capture(frame, kernel, memory_region, nonblocking)