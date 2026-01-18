import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def _get_ZZ_Z_Z_bsa_constructor_args():
    bitstrings = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    chunksizes = np.asarray([4])
    timestamps = np.asarray([datetime.datetime.now()])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    settings = list(cw.observables_to_settings([cirq.Z(a) * cirq.Z(b) * 7, cirq.Z(a) * 5, cirq.Z(b) * 3], qubits=[a, b]))
    meas_spec = _MeasurementSpec(settings[0], {})
    return {'meas_spec': meas_spec, 'simul_settings': settings, 'qubit_to_index': qubit_to_index, 'bitstrings': bitstrings, 'chunksizes': chunksizes, 'timestamps': timestamps}