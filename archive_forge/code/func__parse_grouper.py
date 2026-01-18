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
def _parse_grouper(grouper: Union[str, GROUPER_T]=group_settings_greedy) -> GROUPER_T:
    """Logic for turning a named grouper into one of the build-in groupers in support of the
    high-level `measure_observables` API."""
    if isinstance(grouper, str):
        try:
            grouper = _GROUPING_FUNCS[grouper.lower()]
        except KeyError:
            raise ValueError(f'Unknown grouping function {grouper}')
    return grouper