from __future__ import annotations
from enum import Enum
from typing import NamedTuple, Union, Optional, NewType, Any, List
import numpy as np
from qiskit import pulse
class WaveformType(str, Enum):
    """
    Waveform data type.

    REAL: Assigned to objects that represent real part of waveform.
    IMAG: Assigned to objects that represent imaginary part of waveform.
    OPAQUE: Assigned to objects that represent waveform with unbound parameters.
    """
    REAL = 'Waveform.Real'
    IMAG = 'Waveform.Imag'
    OPAQUE = 'Waveform.Opaque'