import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Type, Union
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.api import QAM, QuantumExecutable, QAMExecutionResult
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.quil import Program
from pyquil.quilatom import Label, LabelPlaceholder, MemoryReference
from pyquil.quilbase import (
def _execute_program(self) -> 'PyQVM':
    self.program_counter = 0
    assert self.program is not None
    halted = len(self.program) == 0
    while not halted:
        halted = self.transition()
    return self