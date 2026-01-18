from __future__ import annotations
import logging
from collections.abc import Callable
from qiskit.providers.backend import Backend
def _match_all(obj, criteria):
    """Return True if all items in criteria matches items in obj."""
    return all((getattr(obj, key_, None) == value_ for key_, value_ in criteria.items()))