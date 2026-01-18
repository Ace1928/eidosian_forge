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
def _needs_init_layer(grouped_settings: Dict[InitObsSetting, List[InitObsSetting]]) -> bool:
    """Helper function to go through init_states and determine if any of them need an
    initialization layer of single-qubit gates."""
    for max_setting in grouped_settings.keys():
        if any((st is not value.KET_ZERO for _, st in max_setting.init_state)):
            return True
    return False