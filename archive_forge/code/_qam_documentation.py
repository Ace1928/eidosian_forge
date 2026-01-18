from typing import Optional, Sequence, TypeVar, Union, cast
import numpy as np
from rpcq.messages import ParameterAref
from pyquil.api._qam import QAM, QAMExecutionResult, QuantumExecutable

        Mutate the provided QAM to add methods and data for backwards compatibility,
        by dynamically mixing in this wrapper class.
        