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
class VarianceStoppingCriteria(StoppingCriteria):
    """Stop sampling when average variance per term drops below a variance bound."""
    variance_bound: float
    repetitions_per_chunk: int = 10000

    def more_repetitions(self, accumulator: BitstringAccumulator) -> int:
        if len(accumulator.bitstrings) == 0:
            return self.repetitions_per_chunk
        cov = accumulator.covariance()
        n_terms = cov.shape[0]
        sum_variance = np.sum(cov)
        var_of_the_e = sum_variance / len(accumulator.bitstrings)
        vpt = var_of_the_e / n_terms
        if vpt <= self.variance_bound:
            return 0
        return self.repetitions_per_chunk

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)