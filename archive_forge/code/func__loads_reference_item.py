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
def _loads_reference_item(type_key, data_bytes, version, metadata_deserializer):
    if type_key == type_keys.Value.NULL:
        return None
    if type_key == type_keys.Program.SCHEDULE_BLOCK:
        return common.data_from_binary(data_bytes, deserializer=read_schedule_block, version=version, metadata_deserializer=metadata_deserializer)
    raise QpyError(f'Loaded schedule reference item is neither None nor ScheduleBlock. Type key {type_key} is not valid data type for a reference items. This data cannot be loaded. Please check QPY version.')