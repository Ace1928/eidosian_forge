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
def _write_symbolic_pulse(file_obj, data, use_symengine):
    class_name_bytes = data.__class__.__name__.encode(common.ENCODE)
    pulse_type_bytes = data.pulse_type.encode(common.ENCODE)
    envelope_bytes = _dumps_symbolic_expr(data.envelope, use_symengine)
    constraints_bytes = _dumps_symbolic_expr(data.constraints, use_symengine)
    valid_amp_conditions_bytes = _dumps_symbolic_expr(data.valid_amp_conditions, use_symengine)
    header_bytes = struct.pack(formats.SYMBOLIC_PULSE_PACK_V2, len(class_name_bytes), len(pulse_type_bytes), len(envelope_bytes), len(constraints_bytes), len(valid_amp_conditions_bytes), data._limit_amplitude)
    file_obj.write(header_bytes)
    file_obj.write(class_name_bytes)
    file_obj.write(pulse_type_bytes)
    file_obj.write(envelope_bytes)
    file_obj.write(constraints_bytes)
    file_obj.write(valid_amp_conditions_bytes)
    common.write_mapping(file_obj, mapping=data._params, serializer=value.dumps_value)
    value.write_value(file_obj, data.duration)
    value.write_value(file_obj, data.name)