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
def _check_nonnegative_timeslot(timeslots: TimeSlots):
    """Test that a channel has no negative timeslots.

    Raises:
        PulseError: If a channel timeslot is negative.
    """
    for chan, chan_timeslots in timeslots.items():
        if chan_timeslots:
            if chan_timeslots[0][0] < 0:
                raise PulseError(f'An instruction on {chan} has a negative starting time.')