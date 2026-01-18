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
def _aggregate_n_repetitions(next_chunk_repetitions: Set[int]) -> int:
    """A stopping criteria can request a different number of more_repetitions for each
    measurement spec. For batching efficiency, we take the max and issue a warning in this case."""
    if len(next_chunk_repetitions) == 1:
        return list(next_chunk_repetitions)[0]
    reps = max(next_chunk_repetitions)
    warnings.warn(f'The stopping criteria specified a various numbers of repetitions to perform next. To be able to submit as a single sweep, the largest value will be used: {reps}.')
    return reps