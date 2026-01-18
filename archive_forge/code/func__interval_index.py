from __future__ import annotations
import abc
import copy
import functools
import itertools
import multiprocessing as mp
import sys
import warnings
from collections.abc import Callable, Iterable
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import rustworkx as rx
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError, UnassignedReferenceError
from qiskit.pulse.instructions import Instruction, Reference
from qiskit.pulse.utils import instruction_duration_validation
from qiskit.pulse.reference_manager import ReferenceManager
from qiskit.utils.multiprocessing import is_main_process
def _interval_index(intervals: list[Interval], interval: Interval) -> int:
    """Find the index of an interval.

    Args:
        intervals: A sorted list of non-overlapping Intervals.
        interval: The interval for which the index into intervals will be found.

    Returns:
        The index of the interval.

    Raises:
        PulseError: If the interval does not exist.
    """
    index = _locate_interval_index(intervals, interval)
    found_interval = intervals[index]
    if found_interval != interval:
        raise PulseError(f'The interval: {interval} does not exist in intervals: {intervals}')
    return index