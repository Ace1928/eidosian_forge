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
def _write_alignment_context(file_obj, context):
    type_key = type_keys.ScheduleAlignment.assign(context)
    common.write_type_key(file_obj, type_key)
    common.write_sequence(file_obj, sequence=context._context_params, serializer=value.dumps_value)