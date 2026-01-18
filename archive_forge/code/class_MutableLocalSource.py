import collections
from enum import Enum
from typing import Any, Callable, Dict, List
from .. import variables
from ..current_scope_id import current_scope_id
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import identity, istype
class MutableLocalSource(Enum):
    """
    If the VariableTracker.mutable_local represents a Variable that:
    - already existed that Dynamo began tracking while introspection (Existing)
    - is a new variable that is created during Dynamo introspection (Local)
    """
    Existing = 0
    Local = 1