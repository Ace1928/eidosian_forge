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
def _convert_parametric_pulse(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
    """Return converted `Play` instruction with parametric pulse operand.

        .. note::

            If parametric pulse label is not provided by the backend, this method naively generates
            a pulse name based on the pulse shape and bound parameters. This pulse name is formatted
            to, for example, `gaussian_a4e3`, here the last four digits are a part of
            the hash string generated based on the pulse shape and the parameters.
            Because we are using a truncated hash for readability,
            there may be a small risk of pulse name collision with other pulses.
            Basically the parametric pulse name is used just for visualization purpose and
            the pulse module should not have dependency on the parametric pulse names.

        Args:
            instruction: Play qobj instruction with parametric pulse

        Yields:
            Qiskit Pulse play schedule instructions
        """
    channel = self.get_channel(instruction.ch)
    try:
        pulse_name = instruction.label
    except AttributeError:
        sorted_params = sorted(instruction.parameters.items(), key=lambda x: x[0])
        base_str = '{pulse}_{params}'.format(pulse=instruction.pulse_shape, params=str(sorted_params))
        short_pulse_id = hashlib.md5(base_str.encode('utf-8')).hexdigest()[:4]
        pulse_name = f'{instruction.pulse_shape}_{short_pulse_id}'
    params = dict(instruction.parameters)
    if 'amp' in params and isinstance(params['amp'], complex):
        params['angle'] = np.angle(params['amp'])
        params['amp'] = np.abs(params['amp'])
    pulse = ParametricPulseShapes.to_type(instruction.pulse_shape)(**params, name=pulse_name)
    yield instructions.Play(pulse, channel)