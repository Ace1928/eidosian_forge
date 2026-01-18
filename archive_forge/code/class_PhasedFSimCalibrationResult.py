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
@dataclasses.dataclass
class PhasedFSimCalibrationResult:
    """The PhasedFSimGate characterization result.

    Attributes:
        parameters: Map from qubit pair to characterization result. For each pair of characterized
            quibts a and b either only (a, b) or only (b, a) is present.
        gate: Characterized gate for each qubit pair. This is copied from the matching
            PhasedFSimCalibrationRequest and is included to preserve execution context.
        options: The options used to gather this result.
        project_id: Google's job project id.
        program_id: Google's job program id.
        job_id: Google's job job id.
    """
    parameters: Dict[Tuple[cirq.Qid, cirq.Qid], PhasedFSimCharacterization]
    gate: cirq.Gate
    options: PhasedFSimCalibrationOptions
    project_id: Optional[str] = None
    program_id: Optional[str] = None
    job_id: Optional[str] = None
    _engine_job: Optional[EngineJob] = None
    _calibration: Optional[Calibration] = None

    def override(self, parameters: PhasedFSimCharacterization) -> 'PhasedFSimCalibrationResult':
        """Creates the new results with certain parameters overridden for all characterizations.

        This functionality can be used to zero-out the corrected angles and do the analysis on
        remaining errors.

        Args:
            parameters: Parameters that will be used when overriding. The angles of that object
                which are not None will be used to replace current parameters for every pair stored.

        Returns:
            New instance of PhasedFSimCalibrationResult with certain parameters overridden.
        """
        return PhasedFSimCalibrationResult(parameters={pair: pair_parameters.override_by(parameters) for pair, pair_parameters in self.parameters.items()}, gate=self.gate, options=self.options)

    def get_parameters(self, a: cirq.Qid, b: cirq.Qid) -> Optional['PhasedFSimCharacterization']:
        """Returns parameters for a qubit pair (a, b) or None when unknown."""
        if (a, b) in self.parameters:
            return self.parameters[a, b]
        elif (b, a) in self.parameters:
            return self.parameters[b, a].parameters_for_qubits_swapped()
        else:
            return None

    @property
    def engine_job(self) -> Optional[EngineJob]:
        """The cirq_google.EngineJob associated with this calibration request.

        Available only when project_id, program_id and job_id attributes are present.
        """
        if self._engine_job is None and self.project_id and self.program_id and self.job_id:
            engine = Engine(project_id=self.project_id)
            self._engine_job = engine.get_program(self.program_id).get_job(self.job_id)
        return self._engine_job

    @property
    def engine_calibration(self) -> Optional[Calibration]:
        """The underlying device calibration that was used for this user-specific calibration.

        This is a cached property that triggers a network call at the first use.
        """
        if self._calibration is None and self.engine_job is not None:
            self._calibration = self.engine_job.get_calibration()
        return self._calibration

    @classmethod
    def _create_parameters_dict(cls, parameters: List[Tuple[cirq.Qid, cirq.Qid, PhasedFSimCharacterization]]) -> Dict[Tuple[cirq.Qid, cirq.Qid], PhasedFSimCharacterization]:
        """Utility function to create parameters from JSON.

        Can be used from child classes to instantiate classes in a _from_json_dict_
        method."""
        return {(q_a, q_b): params for q_a, q_b, params in parameters}

    @classmethod
    def _from_json_dict_(cls, **kwargs) -> 'PhasedFSimCalibrationResult':
        """Magic method for the JSON serialization protocol.

        Converts serialized dictionary into a dict suitable for
        class instantiation."""
        del kwargs['cirq_type']
        kwargs['parameters'] = cls._create_parameters_dict(kwargs['parameters'])
        return cls(**kwargs)

    def _json_dict_(self) -> Dict[str, Any]:
        """Magic method for the JSON serialization protocol."""
        return {'gate': self.gate, 'parameters': [(q_a, q_b, params) for (q_a, q_b), params in self.parameters.items()], 'options': self.options, 'project_id': self.project_id, 'program_id': self.program_id, 'job_id': self.job_id}