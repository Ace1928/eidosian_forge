import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def _get_some_grouped_settings():
    qubits = cirq.LineQubit.range(2)
    q0, q1 = qubits
    terms = [cirq.X(q0), cirq.Y(q1)]
    settings = list(cirq.work.observables_to_settings(terms, qubits))
    grouped_settings = cirq.work.group_settings_greedy(settings)
    return (grouped_settings, qubits)