from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def __delsingleitem__(self, idx=None):
    list.__delitem__(self, idx)
    self.ca.items.pop(idx, None)
    for list_index in sorted(self.ca.items):
        if list_index < idx:
            continue
        self.ca.items[list_index - 1] = self.ca.items.pop(list_index)