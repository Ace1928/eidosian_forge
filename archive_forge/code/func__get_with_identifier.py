import collections
import math
from typing import (
from pip._vendor.resolvelib.providers import AbstractProvider
from .base import Candidate, Constraint, Requirement
from .candidates import REQUIRES_PYTHON_IDENTIFIER
from .factory import Factory
def _get_with_identifier(mapping: Mapping[str, V], identifier: str, default: D) -> Union[D, V]:
    """Get item from a package name lookup mapping with a resolver identifier.

    This extra logic is needed when the target mapping is keyed by package
    name, which cannot be directly looked up with an identifier (which may
    contain requested extras). Additional logic is added to also look up a value
    by "cleaning up" the extras from the identifier.
    """
    if identifier in mapping:
        return mapping[identifier]
    name, open_bracket, _ = identifier.partition('[')
    if open_bracket and name in mapping:
        return mapping[name]
    return default