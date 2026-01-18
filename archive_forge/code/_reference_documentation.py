import warnings
from typing import Any, List, Optional, Sequence, Union
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import P0, P1, KRAUS_OPS, QUANTUM_GATES
from pyquil.simulation.tools import lifted_gate_matrix, lifted_gate, all_bitstrings

        Resets the current state of ReferenceDensitySimulator ``self.density`` to
        ``self.initial_density``.

        :return: ``self`` to support method chaining.
        