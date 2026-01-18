import json
import struct
import zlib
import warnings
from io import BytesIO
import numpy as np
import symengine as sym
from symengine.lib.symengine_wrapper import (  # pylint: disable = no-name-in-module
from qiskit.exceptions import QiskitError
from qiskit.pulse import library, channels, instructions
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.qpy import formats, common, type_keys
from qiskit.qpy.binary_io import value
from qiskit.qpy.exceptions import QpyError
from qiskit.pulse.configuration import Kernel, Discriminator
def _read_symbolic_pulse(file_obj, version):
    make = formats.SYMBOLIC_PULSE._make
    pack = formats.SYMBOLIC_PULSE_PACK
    size = formats.SYMBOLIC_PULSE_SIZE
    header = make(struct.unpack(pack, file_obj.read(size)))
    pulse_type = file_obj.read(header.type_size).decode(common.ENCODE)
    envelope = _loads_symbolic_expr(file_obj.read(header.envelope_size))
    constraints = _loads_symbolic_expr(file_obj.read(header.constraints_size))
    valid_amp_conditions = _loads_symbolic_expr(file_obj.read(header.valid_amp_conditions_size))
    parameters = common.read_mapping(file_obj, deserializer=value.loads_value, version=version, vectors={})
    legacy_library_pulses = ['Gaussian', 'GaussianSquare', 'Drag', 'Constant']
    class_name = 'SymbolicPulse'
    if pulse_type in legacy_library_pulses:
        parameters['angle'] = np.angle(parameters['amp'])
        parameters['amp'] = np.abs(parameters['amp'])
        _amp, _angle = sym.symbols('amp, angle')
        envelope = envelope.subs(_amp, _amp * sym.exp(sym.I * _angle))
        warnings.warn(f'Library pulses with complex amp are no longer supported. {pulse_type} with complex amp was converted to (amp,angle) representation.', UserWarning)
        class_name = 'ScalableSymbolicPulse'
    duration = value.read_value(file_obj, version, {})
    name = value.read_value(file_obj, version, {})
    if class_name == 'SymbolicPulse':
        return library.SymbolicPulse(pulse_type=pulse_type, duration=duration, parameters=parameters, name=name, limit_amplitude=header.amp_limited, envelope=envelope, constraints=constraints, valid_amp_conditions=valid_amp_conditions)
    elif class_name == 'ScalableSymbolicPulse':
        return library.ScalableSymbolicPulse(pulse_type=pulse_type, duration=duration, amp=parameters['amp'], angle=parameters['angle'], parameters=parameters, name=name, limit_amplitude=header.amp_limited, envelope=envelope, constraints=constraints, valid_amp_conditions=valid_amp_conditions)
    else:
        raise NotImplementedError(f"Unknown class '{class_name}'")