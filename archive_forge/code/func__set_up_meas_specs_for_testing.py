import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def _set_up_meas_specs_for_testing():
    q0, q1 = cirq.LineQubit.range(2)
    setting = cw.InitObsSetting(init_state=cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1), observable=cirq.X(q0) * cirq.Y(q1))
    meas_spec = _MeasurementSpec(max_setting=setting, circuit_params={'beta': 0.123, 'gamma': 0.456})
    bsa = cw.BitstringAccumulator(meas_spec, [], {q: i for i, q in enumerate(cirq.LineQubit.range(3))})
    return (bsa, meas_spec)