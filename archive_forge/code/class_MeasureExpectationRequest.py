import re
from contextlib import contextmanager
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import Iterator, Any, Dict, Union, Tuple, Optional, List, cast
import httpx
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping
@dataclass
class MeasureExpectationRequest:
    """
    Request to measure expectations of Pauli operators.
    """
    prep_program: str
    'Quil program to place QVM into a desired state before expectation measurement.'
    pauli_operators: List[str]
    'Quil programs representing Pauli operators for which to measure expectations.'
    seed: Optional[int]
    'PRNG seed. Set this to guarantee repeatable results.'