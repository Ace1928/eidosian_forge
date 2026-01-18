import abc
import itertools
import warnings
from collections import defaultdict
from typing import Union, List
import inspect
import logging
import numpy as np
import pennylane as qml
from pennylane import Device, DeviceError
from pennylane.math import multiply as qmlmul
from pennylane.math import sum as qmlsum
from pennylane.measurements import (
from pennylane.resource import Resources
from pennylane.operation import operation_derivative, Operation
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
def _multi_meas_with_counts_shot_vec(self, circuit: QuantumTape, shot_tuple, r):
    """Auxiliary function of the shot_vec_statistics and execute
        functions for post-processing the results of multiple measurements at
        least one of which was a counts measurement.

        The measurements were executed on a device that defines a shot vector.
        """
    new_r = []
    for idx in range(shot_tuple.copies):
        result_group = []
        for idx2, r_ in enumerate(r):
            measurement_proc = circuit.measurements[idx2]
            if isinstance(measurement_proc, ProbabilityMP) or (isinstance(measurement_proc, SampleMP) and measurement_proc.obs):
                result = r_[:, idx]
            else:
                result = r_[idx]
            if not isinstance(measurement_proc, CountsMP):
                result = self._asarray(result.T)
            result_group.append(result)
        new_r.append(tuple(result_group))
    return new_r