from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
class CommentedMapView(Sized):
    __slots__ = ('_mapping',)

    def __init__(self, mapping):
        self._mapping = mapping

    def __len__(self):
        count = len(self._mapping)
        return count