from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def _yaml_add_comment(self, comment, key=NoComment, value=NoComment):
    """values is set to key to indicate a value attachment of comment"""
    if key is not NoComment:
        self.yaml_key_comment_extend(key, comment)
        return
    if value is not NoComment:
        self.yaml_value_comment_extend(value, comment)
    else:
        self.ca.comment = comment