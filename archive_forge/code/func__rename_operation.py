from __future__ import annotations
import collections.abc
import io
import itertools
import os
import re
import string
from qiskit.circuit import (
from qiskit.circuit.tools import pi_check
from .exceptions import QASM2ExportError
def _rename_operation(operation):
    """Returns the operation with a new name following this pattern: {operation name}_{operation id}"""
    new_name = f'{operation.name}_{id(operation)}'
    updated_operation = operation.copy(name=new_name)
    return updated_operation