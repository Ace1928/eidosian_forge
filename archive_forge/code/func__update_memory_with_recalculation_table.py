from dataclasses import dataclass
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Union, cast
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import ParameterAref, ParameterSpec
from pyquil.api import QuantumExecutable, EncryptedProgram, EngagementManager
from pyquil._memory import Memory
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qpu_client import GetBuffersRequest, QPUClient, BufferResponse, RunProgramRequest
from pyquil.quilatom import (
@classmethod
def _update_memory_with_recalculation_table(cls, program: EncryptedProgram) -> None:
    """
        Update the program's memory with the final values to be patched into the gate parameters,
        according to the arithmetic expressions in the original program.

        For example::

            DECLARE theta REAL
            DECLARE beta REAL
            RZ(3 * theta) 0
            RZ(beta+theta) 0

        gets translated to::

            DECLARE theta REAL
            DECLARE __P REAL[2]
            RZ(__P[0]) 0
            RZ(__P[1]) 0

        and the recalculation table will contain::

            {
                ParameterAref('__P', 0): Mul(3.0, <MemoryReference theta[0]>),
                ParameterAref('__P', 1): Add(<MemoryReference beta[0]>, <MemoryReference theta[0]>)
            }

        Let's say we've made the following two function calls:

        .. code-block:: python

            compiled_program.write_memory(region_name='theta', value=0.5)
            compiled_program.write_memory(region_name='beta', value=0.1)

        After executing this function, our self.variables_shim in the above example would contain
        the following:

        .. code-block:: python

            {
                ParameterAref('theta', 0): 0.5,
                ParameterAref('beta', 0): 0.1,
                ParameterAref('__P', 0): 1.5,       # (3.0) * theta[0]
                ParameterAref('__P', 1): 0.6        # beta[0] + theta[0]
            }

        Once the _variables_shim is filled, execution continues as with regular binary patching.
        """
    assert isinstance(program, EncryptedProgram)
    for mref, expression in program.recalculation_table.items():
        program._memory.values[mref] = float(cls._resolve_memory_references(expression, memory=program._memory))