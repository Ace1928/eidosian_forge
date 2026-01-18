from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
class ChildType(Enum):
    """Enumerator of the child type."""
    MARKDOWN = 'MARKDOWN'
    EXCEPTION = 'EXCEPTION'