from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def _yaml_get_column(self, key):
    column = None
    sel_idx = None
    pre, post = (key - 1, key + 1)
    if pre in self.ca.items:
        sel_idx = pre
    elif post in self.ca.items:
        sel_idx = post
    else:
        for row_idx, _k1 in enumerate(self):
            if row_idx >= key:
                break
            if row_idx not in self.ca.items:
                continue
            sel_idx = row_idx
    if sel_idx is not None:
        column = self._yaml_get_columnX(sel_idx)
    return column