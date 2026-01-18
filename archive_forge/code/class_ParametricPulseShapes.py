import hashlib
import re
import warnings
from enum import Enum
from functools import singledispatchmethod
from typing import Union, List, Iterator, Optional
import numpy as np
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse import channels, instructions, library
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import QiskitError
from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.schedule import Schedule
from qiskit.qobj import QobjMeasurementOption, PulseLibraryItem, PulseQobjInstruction
from qiskit.qobj.utils import MeasLevel
class ParametricPulseShapes(Enum):
    """Map the assembled pulse names to the pulse module waveforms.

    The enum name is the transport layer name for pulse shapes, the
    value is its mapping to the OpenPulse Command in Qiskit.
    """
    gaussian = 'Gaussian'
    gaussian_square = 'GaussianSquare'
    gaussian_square_drag = 'GaussianSquareDrag'
    gaussian_square_echo = 'gaussian_square_echo'
    drag = 'Drag'
    constant = 'Constant'

    @classmethod
    def from_instance(cls, instance: library.SymbolicPulse) -> 'ParametricPulseShapes':
        """Get Qobj name from the pulse class instance.

        Args:
            instance: SymbolicPulse class.

        Returns:
            Qobj name.

        Raises:
            QiskitError: When pulse instance is not recognizable type.
        """
        if isinstance(instance, library.SymbolicPulse):
            return cls(instance.pulse_type)
        raise QiskitError(f"'{instance}' is not valid pulse type.")

    @classmethod
    def to_type(cls, name: str) -> library.SymbolicPulse:
        """Get symbolic pulse class from the name.

        Args:
            name: Qobj name of the pulse.

        Returns:
            Corresponding class.
        """
        return getattr(library, cls[name].value)