import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
@dataclasses.dataclass(frozen=True)
class LocalXEBPhasedFSimCalibrationOptions(XEBPhasedFSimCalibrationOptions):
    """Options for configuring a PhasedFSim calibration using a local version of XEB.

    XEB uses the fidelity of random circuits to characterize PhasedFSim gates. The parameters
    of the gate are varied by a classical optimizer to maximize the observed fidelities.

    These "Local" options (corresponding to `LocalXEBPhasedFSimCalibrationRequest`) instruct
    `cirq_google.run_calibrations` to execute XEB analysis locally (not via the quantum
    engine). As such, `run_calibrations` can work with any `cirq.Sampler`, not just
    `ProcessorSampler`.

    Args:
        n_library_circuits: The number of distinct, two-qubit random circuits to use in our
            library of random circuits. This should be the same order of magnitude as
            `n_combinations`.
        n_combinations: We take each library circuit and randomly assign it to qubit pairs.
            This parameter controls the number of random combinations of the two-qubit random
            circuits we execute. Higher values increase the precision of estimates but linearly
            increase experimental runtime.
        cycle_depths: We run the random circuits at these cycle depths to fit an exponential
            decay in the fidelity.
        fatol: The absolute convergence tolerance for the objective function evaluation in
            the Nelder-Mead optimization. This controls the runtime of the classical
            characterization optimization loop.
        xatol: The absolute convergence tolerance for the parameter estimates in
            the Nelder-Mead optimization. This controls the runtime of the classical
            characterization optimization loop.
        fsim_options: An instance of `XEBPhasedFSimCharacterizationOptions` that controls aspects
            of the PhasedFSim characterization like initial guesses and which angles to
            characterize.
        n_processes: The number of multiprocessing processes to analyze the XEB characterization
            data. By default, we use a value equal to the number of CPU cores. If `1` is specified,
            multiprocessing is not used.
    """
    n_processes: Optional[int] = None

    def create_phased_fsim_request(self, pairs: Tuple[Tuple[cirq.Qid, cirq.Qid], ...], gate: cirq.Gate):
        return LocalXEBPhasedFSimCalibrationRequest(pairs=pairs, gate=gate, options=self)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)