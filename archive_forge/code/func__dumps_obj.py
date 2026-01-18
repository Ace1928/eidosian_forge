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
def _dumps_obj(obj):
    """Wraps `value.dumps_value` to serialize dictionary and list objects
    which are not supported by `value.dumps_value`.
    """
    if isinstance(obj, dict):
        with BytesIO() as container:
            common.write_mapping(file_obj=container, mapping=obj, serializer=_dumps_obj)
            binary_data = container.getvalue()
        return (b'D', binary_data)
    elif isinstance(obj, list):
        with BytesIO() as container:
            common.write_sequence(file_obj=container, sequence=obj, serializer=_dumps_obj)
            binary_data = container.getvalue()
        return (b'l', binary_data)
    else:
        return value.dumps_value(obj)