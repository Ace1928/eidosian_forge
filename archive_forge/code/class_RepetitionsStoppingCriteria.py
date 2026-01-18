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
@dataclasses.dataclass(frozen=True)
class RepetitionsStoppingCriteria(StoppingCriteria):
    """Stop sampling when the number of repetitions has been reached."""
    total_repetitions: int
    repetitions_per_chunk: int = 10000

    def more_repetitions(self, accumulator: BitstringAccumulator) -> int:
        done = accumulator.n_repetitions
        todo = self.total_repetitions - done
        if todo <= 0:
            return 0
        to_do_next = min(self.repetitions_per_chunk, todo)
        return to_do_next

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)