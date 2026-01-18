from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def __setsingleitem__(self, idx, value):
    if idx < len(self):
        if isinstance(value, string_types) and (not isinstance(value, ScalarString)) and isinstance(self[idx], ScalarString):
            value = type(self[idx])(value)
    list.__setitem__(self, idx, value)