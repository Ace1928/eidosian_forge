from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
class TaggedScalar(CommentedBase):

    def __init__(self):
        self.value = None
        self.style = None

    def __str__(self):
        return self.value