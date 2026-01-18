from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
def _create_child(self, type: ChildType, kwargs: Dict[str, Any]) -> None:
    """Create a new child with the given params"""
    if type == ChildType.MARKDOWN:
        self.markdown(**kwargs)
    elif type == ChildType.EXCEPTION:
        self.exception(**kwargs)
    else:
        raise RuntimeError(f'Unexpected child type {type}')