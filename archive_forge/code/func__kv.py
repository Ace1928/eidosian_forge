from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def _kv(self, k, x0, x1):
    if self.data is None:
        return None
    data = self.data[k]
    return (data[x0], data[x1])