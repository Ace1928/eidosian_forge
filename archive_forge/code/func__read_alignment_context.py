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
def _read_alignment_context(file_obj, version):
    type_key = common.read_type_key(file_obj)
    context_params = common.read_sequence(file_obj, deserializer=value.loads_value, version=version, vectors={})
    context_cls = type_keys.ScheduleAlignment.retrieve(type_key)
    instance = object.__new__(context_cls)
    instance._context_params = tuple(context_params)
    return instance