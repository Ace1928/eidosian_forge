from __future__ import annotations
import dataclasses
import enum
import logging
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple
from torch.onnx._internal.diagnostics.infra import formatter, sarif
class PatchedPropertyBag(sarif.PropertyBag):
    """Key/value pairs that provide additional information about the object.

    The definition of PropertyBag via SARIF spec is "A property bag is an object (§3.6)
    containing an unordered set of properties with arbitrary names." However it is not
    reflected in the json file, and therefore not captured by the python representation.
    This patch adds additional **kwargs to the `__init__` method to allow recording
    arbitrary key/value pairs.
    """

    def __init__(self, tags: Optional[List[str]]=None, **kwargs):
        super().__init__(tags=tags)
        self.__dict__.update(kwargs)