from __future__ import annotations
import sys
import copy
from ruamel.yaml.compat import ordereddict
from ruamel.yaml.compat import MutableSliceableSequence, nprintf  # NOQA
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
from ruamel.yaml.tag import Tag
from collections.abc import MutableSet, Sized, Set, Mapping
def _yaml_clear_pre_comment(self) -> Any:
    pre_comments: List[Any] = []
    if self.ca.comment is None:
        self.ca.comment = [None, pre_comments]
    else:
        self.ca.comment[1] = pre_comments
    return pre_comments