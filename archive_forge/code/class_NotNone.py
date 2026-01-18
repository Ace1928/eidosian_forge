from __future__ import annotations
import sys
import copy
from ruamel.yaml.compat import ordereddict
from ruamel.yaml.compat import MutableSliceableSequence, nprintf  # NOQA
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
from ruamel.yaml.tag import Tag
from collections.abc import MutableSet, Sized, Set, Mapping
class NotNone:
    pass