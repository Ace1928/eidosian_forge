from __future__ import annotations
import sys
import copy
from ruamel.yaml.compat import ordereddict
from ruamel.yaml.compat import MutableSliceableSequence, nprintf  # NOQA
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
from ruamel.yaml.tag import Tag
from collections.abc import MutableSet, Sized, Set, Mapping
class IDX:

    def __init__(self) -> None:
        self._idx = 0

    def __call__(self) -> Any:
        x = self._idx
        self._idx += 1
        return x

    def __str__(self) -> Any:
        return str(self._idx)