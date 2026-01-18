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
class FloquetPhasedFSimCalibrationOptions(PhasedFSimCalibrationOptions):
    """Options specific to Floquet PhasedFSimCalibration.

    Some angles require another angle to be characterized first so result might have more angles
    characterized than requested here.

    Attributes:
        characterize_theta: Whether to characterize θ angle.
        characterize_zeta: Whether to characterize ζ angle.
        characterize_chi: Whether to characterize χ angle.
        characterize_gamma: Whether to characterize γ angle.
        characterize_phi: Whether to characterize φ angle.
        readout_error_tolerance: Threshold for pairwise-correlated readout errors above which the
            calibration will report to fail. Just before each calibration all pairwise two-qubit
            readout errors are checked and when any of the pairs reports an error above the
            threshold, the calibration will fail. This value is a sanity check to determine if
            calibration is reasonable and allows for quick termination if it is not. Set to 1.0 to
            disable readout error checks and None to use default, device-specific thresholds.
    """
    characterize_theta: bool
    characterize_zeta: bool
    characterize_chi: bool
    characterize_gamma: bool
    characterize_phi: bool
    readout_error_tolerance: Optional[float] = None
    version: int = 2
    measure_qubits: Optional[Tuple[cirq.Qid, ...]] = None

    def zeta_chi_gamma_correction_override(self) -> PhasedFSimCharacterization:
        """Gives a PhasedFSimCharacterization that can be used to override characterization after
        correcting for zeta, chi and gamma angles.
        """
        return PhasedFSimCharacterization(zeta=0.0 if self.characterize_zeta else None, chi=0.0 if self.characterize_chi else None, gamma=0.0 if self.characterize_gamma else None)

    def create_phased_fsim_request(self, pairs: Tuple[Tuple[cirq.Qid, cirq.Qid], ...], gate: cirq.Gate) -> 'FloquetPhasedFSimCalibrationRequest':
        return FloquetPhasedFSimCalibrationRequest(pairs=pairs, gate=gate, options=self)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)