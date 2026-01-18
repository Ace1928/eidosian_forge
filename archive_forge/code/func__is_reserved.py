from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
@classmethod
def _is_reserved(cls, name: str) -> bool:
    """Check if the given name corresponds to a reserved flag entry."""
    return name.startswith('R') and name[1:].isdigit()