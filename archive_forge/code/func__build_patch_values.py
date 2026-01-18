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
def _build_patch_values(cls, program: EncryptedProgram) -> Dict[str, List[Union[int, float]]]:
    """
        Construct the patch values from the program to be used in execution.
        """
    patch_values = {}
    cls._update_memory_with_recalculation_table(program=program)
    assert isinstance(program, EncryptedProgram)
    recalculation_table = program.recalculation_table
    memory_ref_names = list(set((mr.name for mr in recalculation_table.keys())))
    if memory_ref_names:
        assert len(memory_ref_names) == 1, 'We expected only one declared memory region for the gate parameter arithmetic replacement references.'
        memory_reference_name = memory_ref_names[0]
        patch_values[memory_reference_name] = [0.0] * len(recalculation_table)
    for name, spec in program.memory_descriptors.items():
        if any((name == mref.name for mref in program.ro_sources)):
            continue
        initial_value = 0.0 if spec.type == 'REAL' else 0
        patch_values[name] = [initial_value] * spec.length
    for k, v in program._memory.values.items():
        patch_values[k.name][k.index] = v
    return patch_values