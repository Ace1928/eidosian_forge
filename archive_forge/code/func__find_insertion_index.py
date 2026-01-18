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
def _find_insertion_index(intervals: list[Interval], new_interval: Interval) -> int:
    """Using binary search on start times, return the index into `intervals` where the new interval
    belongs, or raise an error if the new interval overlaps with any existing ones.
    Args:
        intervals: A sorted list of non-overlapping Intervals.
        new_interval: The interval for which the index into intervals will be found.
    Returns:
        The index into intervals that new_interval should be inserted to maintain a sorted list
        of intervals.
    Raises:
        PulseError: If new_interval overlaps with the given intervals.
    """
    index = _locate_interval_index(intervals, new_interval)
    if index < len(intervals):
        if _overlaps(intervals[index], new_interval):
            raise PulseError('New interval overlaps with existing.')
        return index if new_interval[1] <= intervals[index][0] else index + 1
    return index