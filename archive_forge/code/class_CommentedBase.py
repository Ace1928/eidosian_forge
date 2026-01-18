from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
class CommentedBase(object):

    @property
    def ca(self):
        if not hasattr(self, Comment.attrib):
            setattr(self, Comment.attrib, Comment())
        return getattr(self, Comment.attrib)

    def yaml_end_comment_extend(self, comment, clear=False):
        if comment is None:
            return
        if clear or self.ca.end is None:
            self.ca.end = []
        self.ca.end.extend(comment)

    def yaml_key_comment_extend(self, key, comment, clear=False):
        r = self.ca._items.setdefault(key, [None, None, None, None])
        if clear or r[1] is None:
            if comment[1] is not None:
                assert isinstance(comment[1], list)
            r[1] = comment[1]
        else:
            r[1].extend(comment[0])
        r[0] = comment[0]

    def yaml_value_comment_extend(self, key, comment, clear=False):
        r = self.ca._items.setdefault(key, [None, None, None, None])
        if clear or r[3] is None:
            if comment[1] is not None:
                assert isinstance(comment[1], list)
            r[3] = comment[1]
        else:
            r[3].extend(comment[0])
        r[2] = comment[0]

    def yaml_set_start_comment(self, comment, indent=0):
        """overwrites any preceding comment lines on an object
        expects comment to be without `#` and possible have multiple lines
        """
        from .error import CommentMark
        from .tokens import CommentToken
        pre_comments = self._yaml_get_pre_comment()
        if comment[-1] == '\n':
            comment = comment[:-1]
        start_mark = CommentMark(indent)
        for com in comment.split('\n'):
            pre_comments.append(CommentToken('# ' + com + '\n', start_mark, None))

    def yaml_set_comment_before_after_key(self, key, before=None, indent=0, after=None, after_indent=None):
        """
        expects comment (before/after) to be without `#` and possible have multiple lines
        """
        from ruamel.yaml.error import CommentMark
        from ruamel.yaml.tokens import CommentToken

        def comment_token(s, mark):
            return CommentToken(('# ' if s else '') + s + '\n', mark, None)
        if after_indent is None:
            after_indent = indent + 2
        if before and len(before) > 1 and (before[-1] == '\n'):
            before = before[:-1]
        if after and after[-1] == '\n':
            after = after[:-1]
        start_mark = CommentMark(indent)
        c = self.ca.items.setdefault(key, [None, [], None, None])
        if before == '\n':
            c[1].append(comment_token('', start_mark))
        elif before:
            for com in before.split('\n'):
                c[1].append(comment_token(com, start_mark))
        if after:
            start_mark = CommentMark(after_indent)
            if c[3] is None:
                c[3] = []
            for com in after.split('\n'):
                c[3].append(comment_token(com, start_mark))

    @property
    def fa(self):
        """format attribute

        set_flow_style()/set_block_style()"""
        if not hasattr(self, Format.attrib):
            setattr(self, Format.attrib, Format())
        return getattr(self, Format.attrib)

    def yaml_add_eol_comment(self, comment, key=NoComment, column=None):
        """
        there is a problem as eol comments should start with ' #'
        (but at the beginning of the line the space doesn't have to be before
        the #. The column index is for the # mark
        """
        from .tokens import CommentToken
        from .error import CommentMark
        if column is None:
            try:
                column = self._yaml_get_column(key)
            except AttributeError:
                column = 0
        if comment[0] != '#':
            comment = '# ' + comment
        if column is None:
            if comment[0] == '#':
                comment = ' ' + comment
                column = 0
        start_mark = CommentMark(column)
        ct = [CommentToken(comment, start_mark, None), None]
        self._yaml_add_eol_comment(ct, key=key)

    @property
    def lc(self):
        if not hasattr(self, LineCol.attrib):
            setattr(self, LineCol.attrib, LineCol())
        return getattr(self, LineCol.attrib)

    def _yaml_set_line_col(self, line, col):
        self.lc.line = line
        self.lc.col = col

    def _yaml_set_kv_line_col(self, key, data):
        self.lc.add_kv_line_col(key, data)

    def _yaml_set_idx_line_col(self, key, data):
        self.lc.add_idx_line_col(key, data)

    @property
    def anchor(self):
        if not hasattr(self, Anchor.attrib):
            setattr(self, Anchor.attrib, Anchor())
        return getattr(self, Anchor.attrib)

    def yaml_anchor(self):
        if not hasattr(self, Anchor.attrib):
            return None
        return self.anchor

    def yaml_set_anchor(self, value, always_dump=False):
        self.anchor.value = value
        self.anchor.always_dump = always_dump

    @property
    def tag(self):
        if not hasattr(self, Tag.attrib):
            setattr(self, Tag.attrib, Tag())
        return getattr(self, Tag.attrib)

    def yaml_set_tag(self, value):
        self.tag.value = value

    def copy_attributes(self, t, deep=False):
        for a in [Comment.attrib, Format.attrib, LineCol.attrib, Anchor.attrib, Tag.attrib, merge_attrib]:
            if hasattr(self, a):
                if deep:
                    setattr(t, a, copy.deepcopy(getattr(self, a)))
                else:
                    setattr(t, a, getattr(self, a))

    def _yaml_add_eol_comment(self, comment, key):
        raise NotImplementedError

    def _yaml_get_pre_comment(self):
        raise NotImplementedError

    def _yaml_get_column(self, key):
        raise NotImplementedError