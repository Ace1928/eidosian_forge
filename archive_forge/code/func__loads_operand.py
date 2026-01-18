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
def _loads_operand(type_key, data_bytes, version, use_symengine):
    if type_key == type_keys.ScheduleOperand.WAVEFORM:
        return common.data_from_binary(data_bytes, _read_waveform, version=version)
    if type_key == type_keys.ScheduleOperand.SYMBOLIC_PULSE:
        if version < 6:
            return common.data_from_binary(data_bytes, _read_symbolic_pulse, version=version)
        else:
            return common.data_from_binary(data_bytes, _read_symbolic_pulse_v6, version=version, use_symengine=use_symengine)
    if type_key == type_keys.ScheduleOperand.CHANNEL:
        return common.data_from_binary(data_bytes, _read_channel, version=version)
    if type_key == type_keys.ScheduleOperand.OPERAND_STR:
        return data_bytes.decode(common.ENCODE)
    if type_key == type_keys.ScheduleOperand.KERNEL:
        return common.data_from_binary(data_bytes, _read_kernel, version=version)
    if type_key == type_keys.ScheduleOperand.DISCRIMINATOR:
        return common.data_from_binary(data_bytes, _read_discriminator, version=version)
    return value.loads_value(type_key, data_bytes, version, {})