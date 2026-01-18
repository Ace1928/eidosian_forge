from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Mapping, Optional, TypeVar
import numpy as np
from pyquil.api._abstract_compiler import QuantumExecutable
@dataclass
class QAMExecutionResult:
    executable: QuantumExecutable
    'The executable corresponding to this result.'
    readout_data: Mapping[str, Optional[np.ndarray]] = field(default_factory=dict)
    'Readout data returned from the QAM, keyed on the name of the readout register or post-processing node.'
    execution_duration_microseconds: Optional[int] = field(default=None)
    'Duration job held exclusive hardware access. Defaults to ``None`` when information is not available.'