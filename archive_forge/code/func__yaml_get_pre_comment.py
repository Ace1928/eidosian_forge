from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def _yaml_get_pre_comment(self):
    pre_comments = []
    if self.ca.comment is None:
        self.ca.comment = [None, pre_comments]
    else:
        self.ca.comment[1] = pre_comments
    return pre_comments