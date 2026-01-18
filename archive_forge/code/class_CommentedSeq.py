from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
class CommentedSeq(MutableSliceableSequence, list, CommentedBase):
    __slots__ = (Comment.attrib, '_lst')

    def __init__(self, *args, **kw):
        list.__init__(self, *args, **kw)

    def __getsingleitem__(self, idx):
        return list.__getitem__(self, idx)

    def __setsingleitem__(self, idx, value):
        if idx < len(self):
            if isinstance(value, string_types) and (not isinstance(value, ScalarString)) and isinstance(self[idx], ScalarString):
                value = type(self[idx])(value)
        list.__setitem__(self, idx, value)

    def __delsingleitem__(self, idx=None):
        list.__delitem__(self, idx)
        self.ca.items.pop(idx, None)
        for list_index in sorted(self.ca.items):
            if list_index < idx:
                continue
            self.ca.items[list_index - 1] = self.ca.items.pop(list_index)

    def __len__(self):
        return list.__len__(self)

    def insert(self, idx, val):
        """the comments after the insertion have to move forward"""
        list.insert(self, idx, val)
        for list_index in sorted(self.ca.items, reverse=True):
            if list_index < idx:
                break
            self.ca.items[list_index + 1] = self.ca.items.pop(list_index)

    def extend(self, val):
        list.extend(self, val)

    def __eq__(self, other):
        return list.__eq__(self, other)

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

    def __deepcopy__(self, memo):
        res = self.__class__()
        memo[id(self)] = res
        for k in self:
            res.append(copy.deepcopy(k))
            self.copy_attributes(res, deep=True)
        return res

    def __add__(self, other):
        return list.__add__(self, other)

    def sort(self, key=None, reverse=False):
        if key is None:
            tmp_lst = sorted(zip(self, range(len(self))), reverse=reverse)
            list.__init__(self, [x[0] for x in tmp_lst])
        else:
            tmp_lst = sorted(zip(map(key, list.__iter__(self)), range(len(self))), reverse=reverse)
            list.__init__(self, [list.__getitem__(self, x[1]) for x in tmp_lst])
        itm = self.ca.items
        self.ca._items = {}
        for idx, x in enumerate(tmp_lst):
            old_index = x[1]
            if old_index in itm:
                self.ca.items[idx] = itm[old_index]

    def __repr__(self):
        return list.__repr__(self)