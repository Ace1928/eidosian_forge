import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available

    Try to parse text-generation-inference error message and raise HTTPError in any case.

    Args:
        error (`HTTPError`):
            The HTTPError that have been raised.
    