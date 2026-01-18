from typing import Sequence, Union, Optional, Dict, List
from collections import Counter
from copy import deepcopy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.result.result import Result
from qiskit.result.counts import Counts
from qiskit.result.distributions.probability import ProbDistribution
from qiskit.result.distributions.quasi import QuasiDistribution
from qiskit.result.postprocess import _bin_to_hex
from qiskit._accelerate import results as results_rs  # pylint: disable=no-name-in-module
def _adjust_creg_sizes(creg_sizes, indices):
    """Helper to reduce creg_sizes to match indices"""
    new_creg_sizes = [[creg[0], 0] for creg in creg_sizes]
    indices_sort = sorted(indices)
    creg_nums = [x for _, x in creg_sizes]
    creg_limits = [sum(creg_nums[0:x:1]) - 1 for x in range(0, len(creg_nums) + 1)][1:]
    creg_idx = 0
    for ind in indices_sort:
        for idx in range(creg_idx, len(creg_limits)):
            if ind <= creg_limits[idx]:
                creg_idx = idx
                new_creg_sizes[idx][1] += 1
                break
    new_creg_sizes = [creg for creg in new_creg_sizes if creg[1] != 0]
    return new_creg_sizes