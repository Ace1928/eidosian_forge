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
def _loads_symbolic_expr(expr_bytes, use_symengine=False):
    if expr_bytes == b'':
        return None
    expr_bytes = zlib.decompress(expr_bytes)
    if use_symengine:
        return load_basic(expr_bytes)
    else:
        from sympy import parse_expr
        expr_txt = expr_bytes.decode(common.ENCODE)
        expr = parse_expr(expr_txt)
        return sym.sympify(expr)