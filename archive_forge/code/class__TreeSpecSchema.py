import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
@dataclasses.dataclass
class _TreeSpecSchema:
    """
    _TreeSpecSchema is the schema used to serialize the TreeSpec
    It contains the following fields:
    - type: A string name of the type. null for the case of a LeafSpec.
    - context: Any format which is json dumpable
    - children_spec: A list of children serialized specs.
    """
    type: Optional[str]
    context: DumpableContext
    children_spec: List['_TreeSpecSchema']