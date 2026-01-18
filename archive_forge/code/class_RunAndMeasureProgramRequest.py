import re
from contextlib import contextmanager
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import Iterator, Any, Dict, Union, Tuple, Optional, List, cast
import httpx
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping
@dataclass
class RunAndMeasureProgramRequest:
    """
    Request to run and measure a Quil program.
    """
    program: str
    'Quil program to run.'
    qubits: List[int]
    'Qubits to measure.'
    trials: int
    'Number of times to run program and collect results.'
    measurement_noise: Optional[Tuple[float, float, float]]
    'Simulated measurement noise for X, Y, and Z axes.'
    gate_noise: Optional[Tuple[float, float, float]]
    'Simulated gate noise for X, Y, and Z axes.'
    seed: Optional[int]
    'PRNG seed. Set this to guarantee repeatable results.'