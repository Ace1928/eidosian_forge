import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
class BadStopping(StoppingCriteria):

    def more_repetitions(self, accumulator: BitstringAccumulator) -> int:
        return -23