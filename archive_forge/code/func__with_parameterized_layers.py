import abc
import dataclasses
import itertools
import os
import tempfile
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, study, ops, value, protocols
from cirq._doc import document
from cirq.work.observable_grouping import group_settings_greedy, GROUPER_T
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import InitObsSetting, observables_to_settings, _MeasurementSpec
def _with_parameterized_layers(circuit: 'cirq.AbstractCircuit', qubits: Sequence['cirq.Qid'], needs_init_layer: bool) -> 'cirq.Circuit':
    """Return a copy of the input circuit with parameterized single-qubit rotations.

    These rotations flank the circuit: the initial two layers of X and Y gates
    are given parameter names "{qubit}-Xi" and "{qubit}-Yi" and are used
    to set up the initial state. If `needs_init_layer` is False,
    these two layers of gates are omitted.

    The final two layers of X and Y gates are given parameter names
    "{qubit}-Xf" and "{qubit}-Yf" and are use to change the frame of the
    qubit before measurement, effectively measuring in bases other than Z.
    """
    x_beg_mom = circuits.Moment([ops.X(q) ** sympy.Symbol(f'{q}-Xi') for q in qubits])
    y_beg_mom = circuits.Moment([ops.Y(q) ** sympy.Symbol(f'{q}-Yi') for q in qubits])
    x_end_mom = circuits.Moment([ops.X(q) ** sympy.Symbol(f'{q}-Xf') for q in qubits])
    y_end_mom = circuits.Moment([ops.Y(q) ** sympy.Symbol(f'{q}-Yf') for q in qubits])
    meas_mom = circuits.Moment([ops.measure(*qubits, key='z')])
    if needs_init_layer:
        total_circuit = circuits.Circuit([x_beg_mom, y_beg_mom])
        total_circuit += circuit.unfreeze()
    else:
        total_circuit = circuit.unfreeze()
    total_circuit.append([x_end_mom, y_end_mom, meas_mom])
    return total_circuit