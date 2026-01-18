from __future__ import annotations
from collections.abc import Iterator, Sequence
from copy import deepcopy
from enum import Enum
from functools import partial
from itertools import chain
import numpy as np
from qiskit import pulse
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import events, types, drawings, device_info
from qiskit.visualization.pulse_v2.stylesheet import QiskitPulseStyle
def _waveform_loader(self, program: pulse.Waveform | pulse.SymbolicPulse):
    """Load Waveform instance.

        This function is sub-routine of py:method:`load_program`.

        Args:
            program: `Waveform` to draw.
        """
    chart = Chart(parent=self)
    fake_inst = pulse.Play(program, types.WaveformChannel())
    inst_data = types.PulseInstruction(t0=0, dt=self.device.dt, frame=types.PhaseFreqTuple(phase=0, freq=0), inst=fake_inst, is_opaque=program.is_parameterized())
    for gen in self.generator['waveform']:
        obj_generator = partial(gen, formatter=self.formatter, device=self.device)
        for data in obj_generator(inst_data):
            chart.add_data(data)
    self.charts.append(chart)