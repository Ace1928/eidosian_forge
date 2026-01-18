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
def _dumps_operand(operand, use_symengine):
    if isinstance(operand, library.Waveform):
        type_key = type_keys.ScheduleOperand.WAVEFORM
        data_bytes = common.data_to_binary(operand, _write_waveform)
    elif isinstance(operand, library.SymbolicPulse):
        type_key = type_keys.ScheduleOperand.SYMBOLIC_PULSE
        data_bytes = common.data_to_binary(operand, _write_symbolic_pulse, use_symengine=use_symengine)
    elif isinstance(operand, channels.Channel):
        type_key = type_keys.ScheduleOperand.CHANNEL
        data_bytes = common.data_to_binary(operand, _write_channel)
    elif isinstance(operand, str):
        type_key = type_keys.ScheduleOperand.OPERAND_STR
        data_bytes = operand.encode(common.ENCODE)
    elif isinstance(operand, Kernel):
        type_key = type_keys.ScheduleOperand.KERNEL
        data_bytes = common.data_to_binary(operand, _write_kernel)
    elif isinstance(operand, Discriminator):
        type_key = type_keys.ScheduleOperand.DISCRIMINATOR
        data_bytes = common.data_to_binary(operand, _write_discriminator)
    else:
        type_key, data_bytes = value.dumps_value(operand)
    return (type_key, data_bytes)