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
def _extract_memory_regions(memory_descriptors: Dict[str, ParameterSpec], ro_sources: Dict[MemoryReference, str], buffers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    first, *rest = buffers.values()
    num_shots = first.shape[0]

    def alloc(spec: ParameterSpec) -> np.ndarray:
        dtype = {'BIT': np.int64, 'INTEGER': np.int64, 'REAL': np.float64, 'FLOAT': np.float64}
        try:
            return np.ndarray((num_shots, spec.length), dtype=dtype[spec.type])
        except KeyError:
            raise ValueError(f'Unexpected memory type {spec.type}.')
    regions: Dict[str, np.ndarray] = {}
    for mref, key in ro_sources.items():
        if mref.name not in memory_descriptors:
            continue
        elif mref.name not in regions:
            regions[mref.name] = alloc(memory_descriptors[mref.name])
        buf = buffers[key]
        if buf.ndim == 1:
            buf = buf.reshape((num_shots, 1))
        if np.iscomplexobj(buf):
            buf = np.column_stack((buf.real, buf.imag))
        _, width = buf.shape
        end = mref.offset + width
        region_width = memory_descriptors[mref.name].length
        if end > region_width:
            raise ValueError(f'Attempted to fill {mref.name}[{mref.offset}, {end})but the declared region has width {region_width}.')
        regions[mref.name][:, mref.offset:end] = buf
    return regions