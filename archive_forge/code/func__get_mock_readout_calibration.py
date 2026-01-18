import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def _get_mock_readout_calibration(qa_0=90, qa_1=10, qb_0=91, qb_1=9):
    q1_ro = np.array([0] * qa_0 + [1] * qa_1)
    q2_ro = np.array([0] * qb_0 + [1] * qb_1)
    rs = np.random.RandomState(52)
    rs.shuffle(q1_ro)
    rs.shuffle(q2_ro)
    ro_bitstrings = np.vstack((q1_ro, q2_ro)).T
    assert ro_bitstrings.shape == (100, 2)
    chunksizes = np.asarray([100])
    timestamps = np.asarray([datetime.datetime.now()])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    ro_settings = list(cw.observables_to_settings([cirq.Z(a), cirq.Z(b)], qubits=[a, b]))
    ro_meas_spec_setting, = list(cw.observables_to_settings([cirq.Z(a) * cirq.Z(b)], qubits=[a, b]))
    ro_meas_spec = _MeasurementSpec(ro_meas_spec_setting, {})
    ro_bsa = cw.BitstringAccumulator(meas_spec=ro_meas_spec, simul_settings=ro_settings, qubit_to_index=qubit_to_index, bitstrings=ro_bitstrings, chunksizes=chunksizes, timestamps=timestamps)
    return (ro_bsa, ro_settings, ro_meas_spec_setting)