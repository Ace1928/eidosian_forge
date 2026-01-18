from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
class CommentedKeySeq(tuple, CommentedBase):
    """This primarily exists to be able to roundtrip keys that are sequences"""

    def _yaml_add_comment(self, comment, key=NoComment):
        if key is not NoComment:
            self.yaml_key_comment_extend(key, comment)
        else:
            self.ca.comment = comment

    def _yaml_add_eol_comment(self, comment, key):
        self._yaml_add_comment(comment, key=key)

    def _yaml_get_columnX(self, key):
        return self.ca.items[key][0].start_mark.column

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

    def _yaml_get_pre_comment(self):
        pre_comments = []
        if self.ca.comment is None:
            self.ca.comment = [None, pre_comments]
        else:
            self.ca.comment[1] = pre_comments
        return pre_comments