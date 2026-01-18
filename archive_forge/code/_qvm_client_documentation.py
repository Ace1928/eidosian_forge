import re
from contextlib import contextmanager
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import Iterator, Any, Dict, Union, Tuple, Optional, List, cast
import httpx
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping

        Errors should contain a "status" field with a human readable explanation of
        what went wrong as well as a "error_type" field indicating the kind of error that can be mapped
        to a Python type.

        There's a fallback error UnknownApiError for other types of exceptions (network issues, api
        gateway problems, etc.)
        