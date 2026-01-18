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
def _loads_obj(type_key, binary_data, version, vectors):
    """Wraps `value.loads_value` to deserialize binary data to dictionary
    or list objects which are not supported by `value.loads_value`.
    """
    if type_key == b'D':
        with BytesIO(binary_data) as container:
            return common.read_mapping(file_obj=container, deserializer=_loads_obj, version=version, vectors=vectors)
    elif type_key == b'l':
        with BytesIO(binary_data) as container:
            return common.read_sequence(file_obj=container, deserializer=_loads_obj, version=version, vectors=vectors)
    else:
        return value.loads_value(type_key, binary_data, version, vectors)