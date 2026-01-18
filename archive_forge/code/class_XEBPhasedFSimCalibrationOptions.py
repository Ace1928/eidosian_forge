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
class XEBPhasedFSimCalibrationOptions(PhasedFSimCalibrationOptions):
    """Options for configuring a PhasedFSim calibration using XEB.

    XEB uses the fidelity of random circuits to characterize PhasedFSim gates. The parameters
    of the gate are varied by a classical optimizer to maximize the observed fidelities.

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
    """
    n_library_circuits: int = 20
    n_combinations: int = 10
    cycle_depths: Tuple[int, ...] = _DEFAULT_XEB_CYCLE_DEPTHS
    fatol: Optional[float] = 0.005
    xatol: Optional[float] = 0.005
    fsim_options: XEBPhasedFSimCharacterizationOptions = XEBPhasedFSimCharacterizationOptions()

    def to_args(self) -> Dict[str, Any]:
        """Convert this dataclass to an `args` dictionary suitable for sending to the Quantum
        Engine calibration API."""
        args: Dict[str, Any] = {'n_library_circuits': self.n_library_circuits, 'n_combinations': self.n_combinations, 'cycle_depths': '_'.join((f'{cd:d}' for cd in self.cycle_depths))}
        if self.fatol is not None:
            args['fatol'] = self.fatol
        if self.xatol is not None:
            args['xatol'] = self.xatol
        fsim_options = dataclasses.asdict(self.fsim_options)
        fsim_options = {k: v for k, v in fsim_options.items() if v is not None}
        args.update(fsim_options)
        return args

    def create_phased_fsim_request(self, pairs: Tuple[Tuple[cirq.Qid, cirq.Qid], ...], gate: cirq.Gate) -> 'XEBPhasedFSimCalibrationRequest':
        return XEBPhasedFSimCalibrationRequest(pairs=pairs, gate=gate, options=self)

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        del kwargs['cirq_type']
        kwargs['cycle_depths'] = tuple(kwargs['cycle_depths'])
        return cls(**kwargs)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)