import io
import itertools
import math
import re
from warnings import warn
import numpy as np
from qiskit.circuit import Clbit, Qubit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library.standard_gates import SwapGate, XGate, ZGate, RZZGate, U1Gate, PhaseGate
from qiskit.circuit.measure import Measure
from qiskit.circuit.tools.pi_check import pi_check
from .qcstyle import load_style
from ._utils import (
def _add_controls(self, wire_list, ctrlqargs, ctrl_state, col):
    """Add one or more controls to a gate"""
    for index, ctrl_item in enumerate(zip(ctrlqargs, ctrl_state)):
        pos = ctrl_item[0]
        nxt = wire_list[index]
        if wire_list[index] > wire_list[-1]:
            nxt -= 1
            while nxt not in wire_list:
                nxt -= 1
        else:
            nxt += 1
            while nxt not in wire_list:
                nxt += 1
        control = '\\ctrlo' if ctrl_item[1] == '0' else '\\ctrl'
        self._latex[pos][col] = f'{control}' + '{' + str(nxt - wire_list[index]) + '}'