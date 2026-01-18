from __future__ import annotations
import collections.abc
import struct
import uuid
import numpy as np
import symengine
from symengine.lib.symengine_wrapper import (  # pylint: disable = no-name-in-module
from qiskit.circuit import CASE_DEFAULT, Clbit, ClassicalRegister
from qiskit.circuit.classical import expr, types
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement
from qiskit.qpy import common, formats, exceptions, type_keys
def _write_expr_type(file_obj, type_: types.Type):
    if type_.kind is types.Bool:
        file_obj.write(type_keys.ExprType.BOOL)
    elif type_.kind is types.Uint:
        file_obj.write(type_keys.ExprType.UINT)
        file_obj.write(struct.pack(formats.EXPR_TYPE_UINT_PACK, *formats.EXPR_TYPE_UINT(type_.width)))
    else:
        raise exceptions.QpyError(f"unhandled Type object '{type_};")