import re
from contextlib import contextmanager
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import Iterator, Any, Dict, Union, Tuple, Optional, List, cast
import httpx
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping
@dataclass
class RunAndMeasureProgramResponse:
    """
    Program run and measure response.
    """
    results: List[List[int]]
    'Resulting memory region value, a multi-dimensional array of size <trials>-by-<slots>.'