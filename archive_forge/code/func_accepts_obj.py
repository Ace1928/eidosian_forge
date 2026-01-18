from itertools import islice, product
from typing import List
import numpy as np
import pennylane as qml
from pennylane import BasisState, QubitDevice, StatePrep
from pennylane.devices import DefaultQubitLegacy
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.wires import Wires
from ._serialize import QuantumScriptSerializer
from ._version import __version__
def accepts_obj(obj):
    if obj.name == 'QFT':
        return len(obj.wires) < 10
    if obj.name == 'GroverOperator':
        return len(obj.wires) < 13
    is_not_tape = not isinstance(obj, qml.tape.QuantumTape)
    is_supported = getattr(self, 'supports_operation', lambda name: False)(obj.name)
    return is_not_tape and is_supported